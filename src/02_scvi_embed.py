import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SCVI_PARAMS = {"n_latent": 30, "n_layers": 2, "n_hidden": 128}
SCVI_TRAIN_PARAMS = {"max_epochs": 400, "early_stopping": False}
N_NEIGHBORS = 15
N_PCS = 30
RANDOM_SEED = 0
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

EC_SUBTYPE_MARKERS = {
    "arterial":   ["GJA5", "SEMA3G", "CXCL12", "HEY1", "NOTCH4", "EFNB2"],
    "capillary":  ["PLVAP", "CA4", "BTNL9", "RGCC"],
    "venous":     ["ACKR1", "NR2F2", "SELP", "SELE"],
    "lymphatic":  ["LYVE1", "PROX1", "FLT4", "CCL21"],
    "pericyte":   ["RGS5", "PDGFRB", "NOTCH3", "ABCC9"],
}

def set_seeds(seed=0):
    import torch, random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def train_scvi(adata):
    import scvi
    scvi.settings.seed = RANDOM_SEED
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample")
    model = scvi.model.SCVI(adata, **SCVI_PARAMS)
    print("  Training scVI on", adata.n_obs, "cells x", adata.var["highly_variable"].sum(), "HVGs...")
    model.train(**SCVI_TRAIN_PARAMS, accelerator="auto")
    final_loss = model.history["train_loss_epoch"].values[-1]
    print("  Training complete. Final train_loss_epoch:", float(np.array(final_loss).flat[0]))
    adata.obsm["X_scVI"] = model.get_latent_representation()
    print("  Latent shape:", adata.obsm["X_scVI"].shape)
    return adata, model

def build_neighbor_graph(adata):
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.umap(adata)
    print("  Neighbour graph built")
    return adata

def run_leiden_multi_resolution(adata):
    results = {}
    for res in LEIDEN_RESOLUTIONS:
        key = "leiden_" + str(res)
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=RANDOM_SEED, flavor="igraph", n_iterations=2, directed=False)
        n_clusters = adata.obs[key].nunique()
        print("  Resolution", res, ":", n_clusters, "clusters ->", key)
        results[res] = {"key": key, "n_clusters": n_clusters}
    return adata, results

def score_resolution_by_marker_separation(adata, resolution_results):
    rows = []
    for res, info in resolution_results.items():
        key = info["key"]
        n_clusters = info["n_clusters"]
        best_clusters = {}
        for subtype, markers in EC_SUBTYPE_MARKERS.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                continue
            cluster_means = {}
            for cl in adata.obs[key].unique():
                mask = adata.obs[key] == cl
                expr = adata[mask, present].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray()
                cluster_means[cl] = float(expr.mean())
            best_clusters[subtype] = max(cluster_means, key=cluster_means.get)
        n_distinct = len(set(best_clusters.values()))
        n_resolved = len(best_clusters)
        rows.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "n_subtypes_resolved": n_resolved,
            "n_distinct_best_clusters": n_distinct,
            "separation_score": n_distinct / max(n_resolved, 1),
        })
    df = pd.DataFrame(rows).sort_values("separation_score", ascending=False)
    best = df.iloc[0]
    recommended_res = best["resolution"]
    print("\n  Resolution comparison:")
    print(df.to_string(index=False))
    print("\n  Recommended resolution:", recommended_res, "(separation score:", round(float(best["separation_score"]), 2), ",", int(best["n_clusters"]), "clusters)")
    return df, recommended_res

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 02: scVI + Leiden Clustering")
    print("=" * 60)
    vasc_path = os.path.join(PROCESSED_DIR, "adata_vascular.h5ad")
    if not os.path.exists(vasc_path):
        print("ERROR:", vasc_path, "not found. Run 01_load_qc.py first.")
        sys.exit(1)
    print("\n[1/5] Loading vascular subset from", vasc_path, "...")
    adata = sc.read_h5ad(vasc_path)
    print("  Loaded:", adata.n_obs, "cells x", adata.n_vars, "genes")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print("  HVG subset:", adata_hvg.n_obs, "cells x", adata_hvg.n_vars, "genes")
    print("\n[2/5] Training scVI...")
    set_seeds(RANDOM_SEED)
    adata_hvg, model = train_scvi(adata_hvg)
    print("\n[3/5] Building neighbour graph and UMAP...")
    adata_hvg = build_neighbor_graph(adata_hvg)
    print("\n[4/5] Running Leiden at multiple resolutions...")
    adata_hvg, resolution_results = run_leiden_multi_resolution(adata_hvg)
    print("\n[5/5] Selecting best Leiden resolution...")
    res_metrics, recommended_res = score_resolution_by_marker_separation(adata_hvg, resolution_results)
    best_key = "leiden_" + str(recommended_res)
    adata_hvg.obs["leiden"] = adata_hvg.obs[best_key].copy()
    adata_hvg.uns["recommended_leiden_resolution"] = recommended_res
    adata_hvg.uns["n_leiden_clusters"] = adata_hvg.obs["leiden"].nunique()
    adata_hvg.uns["pro_raynaud_clusters"] = []
    out_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    metrics_path = os.path.join(PROCESSED_DIR, "resolution_metrics.csv")
    adata_hvg.write_h5ad(out_path)
    res_metrics.to_csv(metrics_path, index=False)
    print("\n" + "=" * 60)
    print("Script 02 complete.")
    print("  scVI object:", out_path)
    print("  Resolution metrics:", metrics_path)
    print("  Final clusters:", adata_hvg.obs["leiden"].nunique())
    print("=" * 60)

if __name__ == "__main__":
    main()
