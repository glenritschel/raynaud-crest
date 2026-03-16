import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

EC_SUBTYPE_MARKERS = {
    "arterial_EC":   ["GJA5", "SEMA3G", "CXCL12", "HEY1", "NOTCH4", "EFNB2"],
    "capillary_EC":  ["PLVAP", "CA4", "BTNL9", "RGCC", "SLC9A3R2"],
    "venous_EC":     ["ACKR1", "NR2F2", "SELP", "SELE", "VCAM1"],
    "lymphatic_EC":  ["LYVE1", "PROX1", "FLT4", "CCL21", "PDPN"],
    "pericyte":      ["RGS5", "PDGFRB", "NOTCH3", "ABCC9", "DES", "MCAM"],
    "pan_EC":        ["VWF", "PECAM1", "CLDN5", "CDH5", "ENG"],
}
CONFIDENCE_THRESHOLD = 1.2

def score_clusters_by_markers(adata, marker_dict, cluster_key="leiden"):
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    scores = {subtype: [] for subtype in marker_dict}
    for cl in clusters:
        mask = adata.obs[cluster_key] == cl
        for subtype, markers in marker_dict.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                scores[subtype].append(0.0)
                continue
            expr = adata[mask, present].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            scores[subtype].append(float(expr.mean()))
    df = pd.DataFrame(scores, index=clusters)
    df.index.name = "cluster"
    return df

def assign_annotations(score_df):
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
            label, confidence = "unknown", "low"
        elif runner_up_score == 0.0 or best_score >= CONFIDENCE_THRESHOLD * runner_up_score:
            label, confidence = best_subtype, "high"
        else:
            label, confidence = best_subtype + "_mixed", "low"
        annotations.append({
            "cluster": cluster,
            "annotation": label,
            "confidence": confidence,
            "best_score": round(best_score, 4),
            "runner_up_score": round(runner_up_score, 4),
            "pan_EC_score": round(score_df.loc[cluster, "pan_EC"], 4) if "pan_EC" in score_df.columns else 0.0,
        })
    return pd.DataFrame(annotations)

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 03: EC Subtype Annotation")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 02_scvi_embed.py first.")
        sys.exit(1)
    print("\n[1/4] Loading scVI object...")
    adata = sc.read_h5ad(in_path)
    print("  Loaded:", adata.n_obs, "cells,", adata.obs["leiden"].nunique(), "clusters")
    if "norm_log" not in adata.layers:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print("\n[2/4] Scoring clusters against EC subtype markers...")
    score_df = score_clusters_by_markers(adata, EC_SUBTYPE_MARKERS)
    print("  Scored", len(score_df), "clusters")
    print("\n[3/4] Assigning subtype annotations...")
    annotation_df = assign_annotations(score_df)
    counts = adata.obs["leiden"].value_counts().rename("n_cells")
    summary = annotation_df.set_index("cluster").join(counts).sort_values("n_cells", ascending=False)
    print("\n  Cluster annotations:")
    print("  " + "-" * 80)
    for cl, row in summary.iterrows():
        print("  Cluster", cl, "|", row["annotation"], "|", row["confidence"], "|", int(row["n_cells"]), "cells | score:", row["best_score"])
    high_conf = (annotation_df["confidence"] == "high").sum()
    print("\n  High confidence:", high_conf, "| Low confidence:", len(annotation_df) - high_conf)
    print("\n[4/4] Saving annotated object...")
    cluster_to_label = dict(zip(annotation_df["cluster"].astype(str), annotation_df["annotation"]))
    cluster_to_conf = dict(zip(annotation_df["cluster"].astype(str), annotation_df["confidence"]))
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_label)
    adata.obs["annotation_confidence"] = adata.obs["leiden"].map(cluster_to_conf)
    adata.uns["ec_subtype_scores"] = score_df.to_dict()
    out_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    adata.write_h5ad(out_path)
    annotation_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_annotations.csv"), index=False)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_marker_scores.csv"))
    print("\n" + "=" * 60)
    print("Script 03 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
