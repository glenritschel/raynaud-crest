"""
RAYNAUD'S CREST PIPELINE
Script 03: Marker-based EC subtype annotation

Key improvement over calcinosis pipeline:
- Calcinosis labeled all clusters as "fibroblast" (no annotation)
- This script assigns each cluster an EC subtype label based on marker expression
- Subtypes: arterial EC, capillary EC, venous EC, lymphatic EC, pericyte, mixed/unknown
- Annotation is used downstream to focus Raynaud's signature scoring

Reference for marker panel:
  Huang et al. 2024 (SSc EC scRNA-seq, GSE138669 + GSE209635)
  Markers: CLDN5, VWF, PECAM1 (pan-EC); GJA5/SEMA3G (arterial);
           PLVAP/CA4 (capillary); ACKR1/NR2F2 (venous); LYVE1/PROX1 (lymphatic)

Inputs:  data/adata_scvi.h5ad
Outputs: data/adata_annotated.h5ad
         data/cluster_annotations.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── EC subtype marker panel ───────────────────────────────────────────────────
# Source: Huang 2024 SSc EC study + Theranostics 2021 dermal EC atlas
EC_SUBTYPE_MARKERS = {
    "arterial_EC":   ["GJA5", "SEMA3G", "CXCL12", "HEY1", "NOTCH4", "EFNB2"],
    "capillary_EC":  ["PLVAP", "CA4", "BTNL9", "RGCC", "SLC9A3R2"],
    "venous_EC":     ["ACKR1", "NR2F2", "SELP", "SELE", "VCAM1"],
    "lymphatic_EC":  ["LYVE1", "PROX1", "FLT4", "CCL21", "PDPN"],
    "pericyte":      ["RGS5", "PDGFRB", "NOTCH3", "ABCC9", "DES", "MCAM"],
    "pan_EC":        ["VWF", "PECAM1", "CLDN5", "CDH5", "ENG"],
}

# Annotation confidence threshold
# A cluster is assigned a subtype if its score is >= CONFIDENCE_THRESHOLD
# times the second-best subtype score
CONFIDENCE_THRESHOLD = 1.2  # must be 20% better than runner-up


def score_clusters_by_markers(adata, marker_dict, cluster_key="leiden"):
    """
    For each cluster, compute mean normalised expression of each marker set.
    Returns a DataFrame: rows=clusters, columns=subtypes.
    """
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    scores = {subtype: [] for subtype in marker_dict}

    for cl in clusters:
        mask = adata.obs[cluster_key] == cl
        cl_adata = adata[mask]
        for subtype, markers in marker_dict.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                scores[subtype].append(0.0)
                continue
            expr = cl_adata[:, present].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            scores[subtype].append(float(expr.mean()))

    df = pd.DataFrame(scores, index=clusters)
    df.index.name = "cluster"
    return df


def assign_annotations(score_df):
    """
    Assign each cluster a subtype label based on highest marker score.
    Flags low-confidence assignments as 'mixed/unknown'.
    """
    # Exclude pan_EC from subtype assignment (used as QC check separately)
    subtype_cols = [c for c in score_df.columns if c != "pan_EC"]
    subtype_scores = score_df[subtype_cols]

    annotations = []
    for cluster in subtype_scores.index:
        row = subtype_scores.loc[cluster]
        best_subtype = row.idxmax()
        best_score = row.max()
        sorted_scores = row.sort_values(ascending=False)
        runner_up_score = sorted_scores.iloc[1] if len(sorted_scores) > 1 else 0.0

        if best_score == 0.0:
            label = "unknown"
            confidence = "low"
        elif runner_up_score == 0.0 or best_score >= CONFIDENCE_THRESHOLD * runner_up_score:
            label = best_subtype
            confidence = "high"
        else:
            label = f"{best_subtype}_mixed"
            confidence = "low"

        annotations.append({
            "cluster": cluster,
            "annotation": label,
            "confidence": confidence,
            "best_score": round(best_score, 4),
            "runner_up_score": round(runner_up_score, 4),
            "pan_EC_score": round(score_df.loc[cluster, "pan_EC"], 4)
            if "pan_EC" in score_df.columns else 0.0,
        })

    return pd.DataFrame(annotations)


def apply_annotations_to_adata(adata, annotation_df, cluster_key="leiden"):
    """Add cluster annotations to adata.obs."""
    cluster_to_label = dict(
        zip(annotation_df["cluster"].astype(str),
            annotation_df["annotation"])
    )
    cluster_to_confidence = dict(
        zip(annotation_df["cluster"].astype(str),
            annotation_df["confidence"])
    )

    adata.obs["cell_type"] = adata.obs[cluster_key].map(cluster_to_label)
    adata.obs["annotation_confidence"] = adata.obs[cluster_key].map(
        cluster_to_confidence
    )
    return adata


def print_annotation_summary(annotation_df, adata, cluster_key="leiden"):
    """Print a readable annotation summary with cell counts."""
    counts = adata.obs[cluster_key].value_counts().rename("n_cells")
    summary = annotation_df.set_index("cluster").join(counts)
    summary = summary.sort_values("n_cells", ascending=False)

    print("\n  Cluster annotations:")
    print(f"  {'Cluster':<10} {'Annotation':<22} {'Confidence':<12} "
          f"{'N cells':<10} {'Best score':<12} {'Pan-EC score'}")
    print("  " + "-" * 80)
    for cl, row in summary.iterrows():
        print(f"  {str(cl):<10} {row['annotation']:<22} {row['confidence']:<12} "
              f"{int(row['n_cells']):<10} {row['best_score']:<12.4f} "
              f"{row['pan_EC_score']:.4f}")


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 03: EC Subtype Annotation")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    in_path = os.path.join(DATA_DIR, "adata_scvi.h5ad")
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 02_scvi_embed.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading scVI object from {in_path}...")
    adata = sc.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes, "
          f"{adata.obs['leiden'].nunique()} clusters")

    # Use normalised expression if available, otherwise normalise
    if "norm_log" not in adata.layers:
        print("  Normalising expression for marker scoring...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Score clusters ────────────────────────────────────────────────────
    print("\n[2/4] Scoring clusters against EC subtype markers...")
    score_df = score_clusters_by_markers(adata, EC_SUBTYPE_MARKERS)
    print(f"  Scored {len(score_df)} clusters across "
          f"{len(EC_SUBTYPE_MARKERS)} subtype signatures")

    # ── Assign annotations ────────────────────────────────────────────────
    print("\n[3/4] Assigning subtype annotations...")
    annotation_df = assign_annotations(score_df)
    print_annotation_summary(annotation_df, adata)

    high_conf = (annotation_df["confidence"] == "high").sum()
    low_conf = (annotation_df["confidence"] == "low").sum()
    print(f"\n  High confidence: {high_conf} clusters")
    print(f"  Low confidence:  {low_conf} clusters")

    # ── Apply and save ────────────────────────────────────────────────────
    print("\n[4/4] Saving annotated object...")
    adata = apply_annotations_to_adata(adata, annotation_df)

    # Store full score matrix in uns for reference
    adata.uns["ec_subtype_scores"] = score_df.to_dict()

    out_path = os.path.join(DATA_DIR, "adata_annotated.h5ad")
    annot_path = os.path.join(DATA_DIR, "cluster_annotations.csv")

    adata.write_h5ad(out_path)
    annotation_df.to_csv(annot_path, index=False)
    score_df.to_csv(os.path.join(DATA_DIR, "cluster_marker_scores.csv"))

    print("\n" + "=" * 60)
    print("Script 03 complete.")
    print(f"  Annotated object:    {out_path}")
    print(f"  Annotations table:   {annot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
