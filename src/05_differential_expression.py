import os, sys
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
N_TOP_GENES = 150

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 05: Differential Expression")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 04_signature_scoring.py first.")
        sys.exit(1)
    print("\n[1/4] Loading scored object...")
    adata = sc.read_h5ad(in_path)
    n_clusters = adata.obs["leiden"].nunique()
    print("  Loaded:", adata.n_obs, "cells,", n_clusters, "clusters")
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print("\n[2/4] Running Wilcoxon DE...")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=False, key_added="rank_genes_groups", pts=True)
    print("  DE complete:", n_clusters, "clusters x", adata.n_vars, "genes =", n_clusters * adata.n_vars, "pairs")
    print("\n[3/4] Extracting top", N_TOP_GENES, "up + down per cluster...")
    result = adata.uns["rank_genes_groups"]
    rows = []
    for cl in result["names"].dtype.names:
        genes = result["names"][cl]
        scores = result["scores"][cl]
        pvals = result["pvals_adj"][cl]
        cl_df = pd.DataFrame({"cluster": cl, "gene": genes, "score": scores, "pval_adj": pvals})
        top_up = cl_df.nlargest(N_TOP_GENES, "score").copy()
        top_up["direction"] = "up"
        top_down = cl_df.nsmallest(N_TOP_GENES, "score").copy()
        top_down["direction"] = "down"
        rows.extend([top_up, top_down])
    top_genes_df = pd.concat(rows, ignore_index=True)
    print("  Extracted", len(top_genes_df), "gene-cluster pairs")
    pro_clusters = list(adata.uns.get("pro_raynaud_clusters", []))
    if pro_clusters:
        pro_clusters_str = [str(c) for c in pro_clusters]
        adata.obs["pro_raynaud_group"] = adata.obs["leiden"].apply(lambda x: "pro_raynaud" if str(x) in pro_clusters_str else "other")
        print("  Running focused DE: pro-Raynaud clusters", pro_clusters_str, "vs rest...")
        sc.tl.rank_genes_groups(adata, groupby="pro_raynaud_group", groups=["pro_raynaud"], reference="other", method="wilcoxon", use_raw=False, key_added="rank_genes_proraynaud")
        pr_result = adata.uns["rank_genes_proraynaud"]
        pr_df = pd.DataFrame({"gene": pr_result["names"]["pro_raynaud"], "score": pr_result["scores"]["pro_raynaud"], "pval_adj": pr_result["pvals_adj"]["pro_raynaud"]}).sort_values("score", ascending=False)
        pr_df.to_csv(os.path.join(PROCESSED_DIR, "de_proraynaud_vs_rest.csv"), index=False)
        print("  Pro-Raynaud DE saved.")
    else:
        print("  No pro-Raynaud clusters — skipping focused DE.")
    print("\n[4/4] Saving...")
    de_path = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    adata_path = os.path.join(PROCESSED_DIR, "adata_de.h5ad")
    top_genes_df.to_csv(de_path, index=False)
    adata.write_h5ad(adata_path)
    print("\n" + "=" * 60)
    print("Script 05 complete.")
    print("  Top gene lists:", de_path)
    print("  DE object:", adata_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
