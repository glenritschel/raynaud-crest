import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAYNAUD_SIGNATURES = {
    "vasospasm": ["EDNRA", "EDNRB", "EDN1", "EDN2", "ADRA1A", "ADRA2A", "ADRA2B", "ROCK1", "ROCK2", "RHOA", "MYLK", "MYL9", "TBXA2R", "PTGIS"],
    "endothelial_injury": ["ICAM1", "VCAM1", "SELE", "SELP", "VWF", "THBD", "TFPI", "CXCL8", "CCL2", "IL6", "CASP3", "CASP8", "BAX", "HSPG2", "APLNR"],
    "impaired_angiogenesis": ["VEGFA", "VEGFC", "KDR", "FLT1", "FLT4", "ANGPT1", "ANGPT2", "TEK", "DLL4", "NOTCH1", "NRP1", "THBS1", "THBS2", "SERPINF1", "APLNR"],
    "oxidative_stress": ["NOX4", "NOX2", "CYBA", "NCF1", "SOD1", "SOD2", "CAT", "GPX1", "TXN", "HMOX1", "HSPA1A", "NOS3", "NOS1", "HIF1A", "EGLN1", "VEGFA"],
    "pericyte_dysfunction": ["PDGFRB", "PDGFB", "RGS5", "ANGPT2", "TIE1", "NOTCH3", "JAG1", "DLL1", "ACTA2", "MYH11", "CNN1", "TAGLN", "COL4A1", "COL4A2", "LAMA4"],
}

def score_signatures(adata, signatures, cluster_key="leiden"):
    cluster_scores = {sig: [] for sig in signatures}
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    for sig_name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            print("  " + sig_name + ": " + str(len(present)) + "/" + str(len(genes)) + " genes found")
        else:
            print("  " + sig_name + ": all " + str(len(genes)) + " genes found")
        if not present:
            adata.obs["score_" + sig_name] = 0.0
            for _ in clusters:
                cluster_scores[sig_name].append(0.0)
            continue
        expr = adata[:, present].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        cell_scores = expr.mean(axis=1)
        adata.obs["score_" + sig_name] = cell_scores
        for cl in clusters:
            mask = adata.obs[cluster_key] == cl
            cluster_scores[sig_name].append(float(cell_scores[mask].mean()))
    score_df = pd.DataFrame(cluster_scores, index=clusters)
    score_df.index.name = "cluster"
    return adata, score_df

def score_by_cell_type(adata, signatures, cell_type_key="cell_type"):
    rows = []
    for ct in adata.obs[cell_type_key].unique():
        mask = adata.obs[cell_type_key] == ct
        row = {"cell_type": ct, "n_cells": int(mask.sum())}
        for sig_name in signatures:
            score_col = "score_" + sig_name
            row[sig_name] = round(float(adata.obs.loc[mask, score_col].mean()), 4) if score_col in adata.obs.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 04: Signature Scoring")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 03_annotate_clusters.py first.")
        sys.exit(1)
    print("\n[1/4] Loading annotated object...")
    adata = sc.read_h5ad(in_path)
    print("  Loaded:", adata.n_obs, "cells x", adata.n_vars, "genes")
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print("\n[2/4] Scoring Raynaud's gene signatures...")
    adata, score_df = score_signatures(adata, RAYNAUD_SIGNATURES)
    score_df["raynaud_primary_score"] = score_df.get("vasospasm", 0) + score_df.get("endothelial_injury", 0)
    score_df["raynaud_composite_score"] = score_df[[c for c in RAYNAUD_SIGNATURES.keys() if c in score_df.columns]].sum(axis=1)
    top_clusters = score_df.nlargest(3, "raynaud_primary_score")
    print("\n  Top 3 pro-Raynaud clusters:")
    for cl, row in top_clusters.iterrows():
        ct = adata.obs.loc[adata.obs["leiden"] == str(cl), "cell_type"].values[0] if "cell_type" in adata.obs.columns else "unknown"
        print("    Cluster", cl, "(" + ct + "): vasospasm=" + str(round(row.get("vasospasm", 0), 4)) + ", EC_injury=" + str(round(row.get("endothelial_injury", 0), 4)))
    print("\n[3/4] Computing scores by EC subtype...")
    if "cell_type" in adata.obs.columns:
        subtype_scores = score_by_cell_type(adata, RAYNAUD_SIGNATURES)
        print(subtype_scores.round(4).to_string(index=False))
    else:
        subtype_scores = pd.DataFrame()
    print("\n[4/4] Saving...")
    adata.uns["pro_raynaud_clusters"] = top_clusters.index.tolist()
    out_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    adata.write_h5ad(out_path)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "signature_scores.csv"))
    if not subtype_scores.empty:
        subtype_scores.to_csv(os.path.join(PROCESSED_DIR, "signature_scores_bytype.csv"), index=False)
    print("\n" + "=" * 60)
    print("Script 04 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
