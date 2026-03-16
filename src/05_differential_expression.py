"""
RAYNAUD'S CREST PIPELINE
Script 05: Wilcoxon differential expression

Performs cluster-vs-rest Wilcoxon rank-sum DE for each Leiden cluster.
Matched to calcinosis pipeline: top 150 up + top 150 down per cluster.

Improvement over calcinosis:
- Also runs DE for pro-Raynaud's clusters vs non-pro-Raynaud's clusters
  (focused comparison rather than just cluster-vs-rest)
- Saves gene lists in Enrichr-ready format

Inputs:  data/adata_scored.h5ad
Outputs: data/de_results.csv           — all DE results
         data/de_top_genes.csv         — top 150 up + 150 down per cluster
         data/de_proraynaud_vs_rest.csv — pro-Raynaud's clusters vs rest
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_TOP_GENES = 150  # matched to calcinosis pipeline


def run_cluster_de(adata, cluster_key="leiden"):
    """
    Wilcoxon rank-sum DE: each cluster vs all other cells.
    Matched to calcinosis pipeline parameters.
    """
    print(f"  Running Wilcoxon DE for {adata.obs[cluster_key].nunique()} clusters...")

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        use_raw=False,
        key_added="rank_genes_groups",
        pts=True,  # compute fraction of cells expressing gene
    )
    return adata


def extract_top_genes(adata, n_top=N_TOP_GENES, key="rank_genes_groups"):
    """
    Extract top N up- and downregulated genes per cluster.
    Returns DataFrame with columns: cluster, direction, gene, score, pval_adj, pts
    """
    clusters = adata.obs["leiden"].unique()
    rows = []

    result = adata.uns[key]
    gene_names = result["names"]
    scores = result["scores"]
    pvals_adj = result["pvals_adj"]
    # pts may not always be present
    pts = result.get("pts", None)

    for cl in gene_names.dtype.names:
        genes = gene_names[cl]
        sc_arr = scores[cl]
        pv_arr = pvals_adj[cl]

        # Build full DE table for this cluster
        cl_df = pd.DataFrame({
            "cluster": cl,
            "gene": genes,
            "score": sc_arr,
            "pval_adj": pv_arr,
        })

        # Top N upregulated (highest positive scores)
        top_up = cl_df.nlargest(n_top, "score").copy()
        top_up["direction"] = "up"

        # Top N downregulated (most negative scores)
        top_down = cl_df.nsmallest(n_top, "score").copy()
        top_down["direction"] = "down"

        rows.append(top_up)
        rows.append(top_down)

    return pd.concat(rows, ignore_index=True)


def run_proraynaud_vs_rest(adata, pro_clusters):
    """
    Additional DE: pro-Raynaud's clusters combined vs all other clusters.
    Improvement over calcinosis — focused comparison for Raynaud's biology.
    """
    if not pro_clusters:
        print("  No pro-Raynaud's clusters identified — skipping focused DE.")
        return pd.DataFrame()

    pro_clusters_str = [str(c) for c in pro_clusters]
    adata.obs["pro_raynaud_group"] = adata.obs["leiden"].apply(
        lambda x: "pro_raynaud" if str(x) in pro_clusters_str else "other"
    )

    print(f"  Running focused DE: pro-Raynaud's clusters "
          f"({pro_clusters_str}) vs rest...")

    sc.tl.rank_genes_groups(
        adata,
        groupby="pro_raynaud_group",
        groups=["pro_raynaud"],
        reference="other",
        method="wilcoxon",
        use_raw=False,
        key_added="rank_genes_proraynaud",
    )

    result = adata.uns["rank_genes_proraynaud"]
    genes = result["names"]["pro_raynaud"]
    scores = result["scores"]["pro_raynaud"]
    pvals = result["pvals_adj"]["pro_raynaud"]

    df = pd.DataFrame({
        "gene": genes,
        "score": scores,
        "pval_adj": pvals,
    }).sort_values("score", ascending=False)

    n_up = (df["score"] > 0).sum()
    n_down = (df["score"] < 0).sum()
    print(f"  Pro-Raynaud's vs rest: {n_up} up, {n_down} down")

    return df


def format_for_enrichr(top_genes_df):
    """
    Format top gene lists for Enrichr submission.
    Returns dict: {cluster_up: [genes], cluster_down: [genes], ...}
    """
    gene_lists = {}
    for cl in top_genes_df["cluster"].unique():
        for direction in ["up", "down"]:
            mask = (top_genes_df["cluster"] == cl) & \
                   (top_genes_df["direction"] == direction)
            genes = top_genes_df.loc[mask, "gene"].tolist()
            key = f"cluster_{cl}_{direction}"
            gene_lists[key] = genes
    return gene_lists


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 05: Differential Expression")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    in_path = os.path.join(DATA_DIR, "adata_scored.h5ad")
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 04_signature_scoring.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading scored object from {in_path}...")
    adata = sc.read_h5ad(in_path)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Loaded: {adata.n_obs} cells, {n_clusters} clusters")

    # Use normalised expression for DE
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Cluster-vs-rest DE ────────────────────────────────────────────────
    print("\n[2/4] Running cluster-vs-rest Wilcoxon DE...")
    adata = run_cluster_de(adata)
    total_pairs = n_clusters * adata.n_vars
    print(f"  DE complete: {n_clusters} clusters × {adata.n_vars} genes "
          f"= {total_pairs:,} gene-cluster pairs")

    # ── Extract top genes ─────────────────────────────────────────────────
    print(f"\n[3/4] Extracting top {N_TOP_GENES} up + {N_TOP_GENES} down per cluster...")
    top_genes_df = extract_top_genes(adata, n_top=N_TOP_GENES)
    print(f"  Extracted {len(top_genes_df):,} gene-cluster pairs "
          f"({n_clusters} clusters × {2 * N_TOP_GENES} genes)")

    # ── Pro-Raynaud's focused DE ──────────────────────────────────────────
    pro_clusters = list(adata.uns.get("pro_raynaud_clusters", []))
    proraynaud_df = run_proraynaud_vs_rest(adata, pro_clusters)

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[4/4] Saving DE results...")
    de_path = os.path.join(DATA_DIR, "de_top_genes.csv")
    pr_path = os.path.join(DATA_DIR, "de_proraynaud_vs_rest.csv")
    adata_path = os.path.join(DATA_DIR, "adata_de.h5ad")

    top_genes_df.to_csv(de_path, index=False)
    if not proraynaud_df.empty:
        proraynaud_df.to_csv(pr_path, index=False)
    adata.write_h5ad(adata_path)

    # Store gene lists in uns for script 06
    adata.uns["enrichr_gene_lists"] = format_for_enrichr(top_genes_df)

    print("\n" + "=" * 60)
    print("Script 05 complete.")
    print(f"  Top gene lists:      {de_path}")
    if not proraynaud_df.empty:
        print(f"  Pro-Raynaud's DE:    {pr_path}")
    print(f"  DE object:           {adata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
