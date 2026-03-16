"""
RAYNAUD'S CREST PIPELINE
Script 04: Raynaud's signature scoring

Scores each cluster against 5 Raynaud's-relevant gene signatures:
  1. Vasospasm / vasoconstriction
  2. Endothelial activation / injury
  3. Impaired angiogenesis
  4. Oxidative stress / NO dysregulation
  5. Pericyte dysfunction

Improvement over calcinosis: scores are also computed per EC subtype
(arterial, capillary, venous) to identify which vascular compartment
drives each pathological programme.

Inputs:  data/adata_annotated.h5ad
Outputs: data/signature_scores.csv      — per-cluster mean scores
         data/signature_scores_bytype.csv — per-subtype mean scores
         data/adata_scored.h5ad
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Raynaud's gene signatures ─────────────────────────────────────────────────
RAYNAUD_SIGNATURES = {
    "vasospasm": [
        # Endothelin system (vasoconstriction)
        "EDNRA", "EDNRB", "EDN1", "EDN2",
        # Adrenergic receptors
        "ADRA1A", "ADRA2A", "ADRA2B",
        # Rho kinase pathway
        "ROCK1", "ROCK2", "RHOA",
        # Myosin light chain kinase
        "MYLK", "MYL9",
        # Thromboxane / prostanoids
        "TBXA2R", "PTGIS",
    ],
    "endothelial_injury": [
        # Adhesion molecules
        "ICAM1", "VCAM1", "SELE", "SELP",
        # Coagulation / vWF
        "VWF", "THBD", "TFPI",
        # Endothelial activation markers
        "CXCL8", "CCL2", "IL6",
        # Apoptosis / injury
        "CASP3", "CASP8", "BAX",
        # EC injury markers from Apostolidis 2018
        "HSPG2", "APLNR",
    ],
    "impaired_angiogenesis": [
        # VEGF signalling
        "VEGFA", "VEGFC", "KDR", "FLT1", "FLT4",
        # Angiopoietin system
        "ANGPT1", "ANGPT2", "TEK",
        # Notch/DLL4 (tip cell regulation)
        "DLL4", "NOTCH1", "NRP1",
        # Anti-angiogenic (upregulated in SSc)
        "THBS1", "THBS2", "SERPINF1",
        # APLNR (impaired in SSc ECs — Apostolidis 2018)
        "APLNR",
    ],
    "oxidative_stress": [
        # NOX family (ROS generation)
        "NOX4", "NOX2", "CYBA", "NCF1",
        # Antioxidant enzymes
        "SOD1", "SOD2", "CAT", "GPX1", "TXN",
        # Heme oxygenase / heat shock
        "HMOX1", "HSPA1A",
        # eNOS / NO production (impaired in SSc)
        "NOS3", "NOS1",
        # HIF-1 pathway
        "HIF1A", "EGLN1", "VEGFA",
    ],
    "pericyte_dysfunction": [
        # Pericyte identity / coverage
        "PDGFRB", "PDGFB", "RGS5",
        # Pericyte detachment markers
        "ANGPT2", "TIE1",
        # Notch signalling (pericyte-EC crosstalk)
        "NOTCH3", "JAG1", "DLL1",
        # Contractility
        "ACTA2", "MYH11", "CNN1", "TAGLN",
        # Extracellular matrix
        "COL4A1", "COL4A2", "LAMA4",
    ],
}


def score_signatures(adata, signatures, cluster_key="leiden"):
    """
    Score each cell against each signature (mean normalised expression).
    Adds per-cell scores to adata.obs.
    Returns per-cluster mean score DataFrame.
    """
    cluster_scores = {sig: [] for sig in signatures}
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))

    for sig_name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]

        if missing:
            print(f"  {sig_name}: {len(present)}/{len(genes)} genes found "
                  f"(missing: {missing[:3]}{'...' if len(missing) > 3 else ''})")
        else:
            print(f"  {sig_name}: all {len(genes)} genes found")

        if not present:
            # All genes missing — score is zero everywhere
            adata.obs[f"score_{sig_name}"] = 0.0
            for _ in clusters:
                cluster_scores[sig_name].append(0.0)
            continue

        # Per-cell score
        expr = adata[:, present].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        cell_scores = expr.mean(axis=1)
        adata.obs[f"score_{sig_name}"] = cell_scores

        # Per-cluster mean
        for cl in clusters:
            mask = adata.obs[cluster_key] == cl
            cluster_scores[sig_name].append(float(cell_scores[mask].mean()))

    score_df = pd.DataFrame(cluster_scores, index=clusters)
    score_df.index.name = "cluster"
    return adata, score_df


def score_by_cell_type(adata, signatures, cell_type_key="cell_type"):
    """
    Compute mean signature scores per annotated EC subtype.
    Improvement over calcinosis: adds biological interpretation by subtype.
    """
    cell_types = adata.obs[cell_type_key].unique()
    rows = []

    for ct in cell_types:
        mask = adata.obs[cell_type_key] == ct
        n = mask.sum()
        row = {"cell_type": ct, "n_cells": int(n)}
        for sig_name in signatures:
            score_col = f"score_{sig_name}"
            if score_col in adata.obs.columns:
                row[sig_name] = round(float(adata.obs.loc[mask, score_col].mean()), 4)
            else:
                row[sig_name] = 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)


def identify_pro_raynaud_clusters(score_df, top_n=3):
    """
    Identify the clusters most likely to be driving Raynaud's pathology.
    Ranks by combined vasospasm + endothelial_injury score (primary disease mechanisms).
    """
    score_df = score_df.copy()
    score_df["raynaud_primary_score"] = (
        score_df.get("vasospasm", 0) +
        score_df.get("endothelial_injury", 0)
    )
    score_df["raynaud_composite_score"] = score_df[
        [c for c in RAYNAUD_SIGNATURES.keys() if c in score_df.columns]
    ].sum(axis=1)

    top = score_df.nlargest(top_n, "raynaud_primary_score")
    return score_df, top


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 04: Signature Scoring")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    in_path = os.path.join(DATA_DIR, "adata_annotated.h5ad")
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 03_annotate_clusters.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading annotated object from {in_path}...")
    adata = sc.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Ensure we use normalised expression
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Score signatures ──────────────────────────────────────────────────
    print("\n[2/4] Scoring Raynaud's gene signatures...")
    adata, score_df = score_signatures(adata, RAYNAUD_SIGNATURES)

    score_df, top_clusters = identify_pro_raynaud_clusters(score_df)

    print("\n  Per-cluster signature scores (top clusters by Raynaud's primary score):")
    cols = list(RAYNAUD_SIGNATURES.keys()) + ["raynaud_primary_score"]
    print(score_df[cols].sort_values("raynaud_primary_score", ascending=False)
          .head(8).round(4).to_string())

    print(f"\n  Top {len(top_clusters)} pro-Raynaud's clusters:")
    for cl, row in top_clusters.iterrows():
        cell_type = adata.obs.loc[
            adata.obs["leiden"] == str(cl), "cell_type"
        ].values[0] if "cell_type" in adata.obs.columns else "unknown"
        print(f"    Cluster {cl} ({cell_type}): "
              f"vasospasm={row.get('vasospasm', 0):.4f}, "
              f"EC_injury={row.get('endothelial_injury', 0):.4f}, "
              f"primary_score={row['raynaud_primary_score']:.4f}")

    # ── Score by cell type ────────────────────────────────────────────────
    print("\n[3/4] Computing scores by EC subtype...")
    if "cell_type" in adata.obs.columns:
        subtype_scores = score_by_cell_type(adata, RAYNAUD_SIGNATURES)
        print("\n  Mean signature scores by EC subtype:")
        print(subtype_scores.round(4).to_string(index=False))
    else:
        subtype_scores = pd.DataFrame()
        print("  (cell_type not found in obs — skipping subtype breakdown)")

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[4/4] Saving scored object...")
    out_path = os.path.join(DATA_DIR, "adata_scored.h5ad")
    scores_path = os.path.join(DATA_DIR, "signature_scores.csv")
    subtype_path = os.path.join(DATA_DIR, "signature_scores_bytype.csv")

    adata.write_h5ad(out_path)
    score_df.to_csv(scores_path)
    if not subtype_scores.empty:
        subtype_scores.to_csv(subtype_path, index=False)

    # Store top cluster IDs for downstream use
    adata.uns["pro_raynaud_clusters"] = top_clusters.index.tolist()

    print("\n" + "=" * 60)
    print("Script 04 complete.")
    print(f"  Scored object:       {out_path}")
    print(f"  Cluster scores:      {scores_path}")
    if not subtype_scores.empty:
        print(f"  Subtype scores:      {subtype_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
