"""
RAYNAUD'S CREST PIPELINE
Script 02: scVI embedding and multi-resolution Leiden clustering

Key improvements over calcinosis pipeline:
- Tests three Leiden resolutions (0.5, 0.8, 1.2) and selects best
- Best resolution = the one that maximally separates known EC subtype markers
- Saves resolution comparison metrics to help interpret the choice
- scVI parameters matched to calcinosis for consistency

Inputs:  data/adata_vascular.h5ad
Outputs: data/adata_scvi.h5ad        — embedded + clustered vascular object
         data/resolution_metrics.csv  — cluster quality by resolution
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── scVI parameters — matched to calcinosis pipeline ─────────────────────────
SCVI_PARAMS = {
    "n_latent": 30,
    "n_layers": 2,
    "n_hidden": 128,
}
SCVI_TRAIN_PARAMS = {
    "max_epochs": 400,
    "early_stopping": False,
}
N_NEIGHBORS = 15
N_PCS = 30
RANDOM_SEED = 0

# Leiden resolutions to test — improvement over calcinosis (single resolution)
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

# Known EC subtype markers for resolution selection
# Higher resolution is better if it separates these marker sets into distinct clusters
EC_SUBTYPE_MARKERS = {
    "arterial":   ["GJA5", "SEMA3G", "CXCL12", "HEY1", "NOTCH4"],
    "capillary":  ["PLVAP", "CA4", "BTNL9", "RGCC"],
    "venous":     ["ACKR1", "NR2F2", "SELP", "SELE"],
    "lymphatic":  ["LYVE1", "PROX1", "FLT4", "CCL21"],
    "pericyte":   ["RGS5", "PDGFRB", "NOTCH3", "ABCC9"],
}


def set_seeds(seed=0):
    import torch
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_scvi(adata):
    """Train scVI variational autoencoder on vascular subset."""
    import scvi
    scvi.settings.seed = RANDOM_SEED

    # scVI requires raw counts
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="sample"
    )

    model = scvi.model.SCVI(adata, **SCVI_PARAMS)

    print(f"  Training scVI on {adata.n_obs} cells × "
          f"{adata.var['highly_variable'].sum()} HVGs...")
    print(f"  accelerator=auto (GPU if available, CPU fallback)")

    model.train(
        **SCVI_TRAIN_PARAMS,
        accelerator="auto",
    )

    final_loss = model.history["train_loss_epoch"].values[-1]
    print(f"  Training complete. Final train_loss_epoch: {final_loss:.2f}")

    # Store latent representation
    adata.obsm["X_scVI"] = model.get_latent_representation()
    print(f"  Latent representation shape: {adata.obsm['X_scVI'].shape}")

    return adata, model


def build_neighbor_graph(adata):
    """Build nearest-neighbour graph on scVI latent space."""
    sc.pp.neighbors(
        adata,
        use_rep="X_scVI",
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS
    )
    sc.tl.umap(adata)
    print(f"  Neighbour graph built (n_neighbors={N_NEIGHBORS})")
    return adata


def run_leiden_multi_resolution(adata):
    """
    Run Leiden clustering at multiple resolutions.
    Improvement over calcinosis: tests 3 resolutions rather than fixing one.
    """
    results = {}
    for res in LEIDEN_RESOLUTIONS:
        key = f"leiden_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=RANDOM_SEED)
        n_clusters = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_clusters} clusters → obs key '{key}'")
        results[res] = {"key": key, "n_clusters": n_clusters}
    return adata, results


def score_resolution_by_marker_separation(adata, resolution_results):
    """
    Score each resolution by how well it separates known EC subtype markers.

    Metric: for each EC subtype, find the cluster with highest mean expression
    of that subtype's markers. A good resolution maximally separates subtypes
    into distinct clusters (different best-cluster for each subtype).

    Returns a DataFrame with resolution scores and recommended resolution.
    """
    rows = []
    for res, info in resolution_results.items():
        key = info["key"]
        n_clusters = info["n_clusters"]

        # Score each subtype marker set per cluster
        best_clusters = {}
        for subtype, markers in EC_SUBTYPE_MARKERS.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                continue
            # Mean expression per cluster
            cluster_means = {}
            for cl in adata.obs[key].unique():
                mask = adata.obs[key] == cl
                expr = adata[mask, present].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray()
                cluster_means[cl] = float(expr.mean())
            best_clusters[subtype] = max(cluster_means, key=cluster_means.get)

        # Count distinct best clusters across subtypes
        n_distinct = len(set(best_clusters.values()))
        n_subtypes_resolved = len(best_clusters)

        rows.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "n_subtypes_resolved": n_subtypes_resolved,
            "n_distinct_best_clusters": n_distinct,
            "separation_score": n_distinct / max(n_subtypes_resolved, 1),
        })

    df = pd.DataFrame(rows).sort_values("separation_score", ascending=False)

    # Recommend: highest separation score, tiebreak by fewer clusters
    best = df.iloc[0]
    recommended_res = best["resolution"]
    print(f"\n  Resolution comparison:")
    print(df.to_string(index=False))
    print(f"\n  Recommended resolution: {recommended_res} "
          f"(separation score: {best['separation_score']:.2f}, "
          f"{int(best['n_clusters'])} clusters)")

    return df, recommended_res


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 02: scVI + Leiden Clustering")
    print("=" * 60)

    # ── Load vascular subset ──────────────────────────────────────────────
    vasc_path = os.path.join(DATA_DIR, "adata_vascular.h5ad")
    if not os.path.exists(vasc_path):
        print(f"ERROR: {vasc_path} not found. Run 01_load_qc.py first.")
        sys.exit(1)

    print(f"\n[1/5] Loading vascular subset from {vasc_path}...")
    adata = sc.read_h5ad(vasc_path)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subset to HVGs for scVI
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print(f"  HVG subset: {adata_hvg.n_obs} cells × {adata_hvg.n_vars} genes")

    # ── Train scVI ────────────────────────────────────────────────────────
    print("\n[2/5] Training scVI...")
    set_seeds(RANDOM_SEED)
    adata_hvg, model = train_scvi(adata_hvg)

    # ── Build neighbour graph ─────────────────────────────────────────────
    print("\n[3/5] Building neighbour graph and UMAP...")
    adata_hvg = build_neighbor_graph(adata_hvg)

    # ── Multi-resolution Leiden ───────────────────────────────────────────
    print("\n[4/5] Running Leiden at multiple resolutions...")
    adata_hvg, resolution_results = run_leiden_multi_resolution(adata_hvg)

    # ── Select best resolution ────────────────────────────────────────────
    print("\n[5/5] Selecting best Leiden resolution...")
    res_metrics, recommended_res = score_resolution_by_marker_separation(
        adata_hvg, resolution_results
    )

    # Add recommended leiden as primary clustering
    best_key = f"leiden_{recommended_res}"
    adata_hvg.obs["leiden"] = adata_hvg.obs[best_key].copy()
    adata_hvg.uns["recommended_leiden_resolution"] = recommended_res
    adata_hvg.uns["n_leiden_clusters"] = adata_hvg.obs["leiden"].nunique()

    print(f"\n  Primary clustering: 'leiden' = leiden_{recommended_res} "
          f"({adata_hvg.obs['leiden'].nunique()} clusters)")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "adata_scvi.h5ad")
    metrics_path = os.path.join(DATA_DIR, "resolution_metrics.csv")

    adata_hvg.write_h5ad(out_path)
    res_metrics.to_csv(metrics_path, index=False)

    print("\n" + "=" * 60)
    print("Script 02 complete.")
    print(f"  scVI object:         {out_path}")
    print(f"  Resolution metrics:  {metrics_path}")
    print(f"  Final clusters:      {adata_hvg.obs['leiden'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
