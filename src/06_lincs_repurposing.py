import os, sys, time, re
import numpy as np
import pandas as pd
import gseapy as gp
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

N_TOP_GENES = 150
TOP_PER_CLUSTER = 15
ENRICHR_DELAY = 1.0
ENRICHR_LIBRARIES = [
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Chem_Pert_down",
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]

def clean_compound_name(term):
    """
    Extract compound name from LINCS term.
    Handles formats:
      LJP006 HME1 24H-PD-0325901-0.37  -> PD-0325901
      CompoundName_CellLine_Dose_Time   -> CompoundName
    """
    # Format: LJPxxx CellLine Time-CompoundName-Dose
    m = re.match(r'^LJP\d+\s+\S+\s+\S+?-(.+)-[\d.]+$', term.strip())
    if m:
        return m.group(1).strip()
    # Fallback: split on underscore
    parts = term.split("_")
    return parts[0].strip() if parts else term.strip()

def run_enrichr_for_cluster(cluster_id, up_genes, down_genes):
    results = []
    for direction, genes, reversal_lib in [
        ("up",   up_genes,   "LINCS_L1000_Chem_Pert_down"),
        ("down", down_genes, "LINCS_L1000_Chem_Pert_up"),
    ]:
        if not genes:
            continue
        for lib in ENRICHR_LIBRARIES:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, outdir=None, verbose=False)
                df = enr.results.copy()
                if df.empty:
                    continue
                df["cluster"] = cluster_id
                df["query_direction"] = direction
                df["library"] = lib
                if lib in ("LINCS_L1000_Chem_Pert_up", "LINCS_L1000_Chem_Pert_down"):
                    adj_p = df["Adjusted P-value"].clip(lower=1e-300)
                    sign = 1.0 if lib == reversal_lib else -1.0
                    df["reversal_score"] = sign * (-np.log10(adj_p))
                    df["compound"] = df["Term"].apply(clean_compound_name)
                else:
                    df["reversal_score"] = 0.0
                    df["compound"] = df["Term"]
                results.append(df)
                time.sleep(ENRICHR_DELAY)
            except Exception as e:
                print("    WARNING: Enrichr failed for cluster", cluster_id, lib, direction, ":", e)
                time.sleep(ENRICHR_DELAY * 2)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def deduplicate_and_rank(raw_df):
    if raw_df.empty:
        return pd.DataFrame()
    lincs_mask = raw_df["library"].str.startswith("LINCS_L1000")
    lincs_df = raw_df[lincs_mask & (raw_df["reversal_score"] > 0)].copy()
    if lincs_df.empty:
        print("  WARNING: No positive LINCS reversal scores found.")
        return pd.DataFrame()
    top_per_cl = lincs_df.sort_values("reversal_score", ascending=False).groupby("cluster").head(TOP_PER_CLUSTER)
    compound_agg = top_per_cl.groupby("compound").agg(
        max_reversal_score=("reversal_score", "max"),
        n_clusters=("cluster", "nunique"),
        clusters=("cluster", lambda x: ",".join(sorted(set(x.astype(str))))),
        best_cluster=("cluster", lambda x: x.loc[x.index[top_per_cl.loc[x.index, "reversal_score"].argmax()]]),
    ).reset_index()
    return compound_agg.sort_values("max_reversal_score", ascending=False)

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 06: LINCS L1000 Reversal Scoring")
    print("=" * 60)
    de_path = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    if not os.path.exists(de_path):
        print("ERROR:", de_path, "not found. Run 05_differential_expression.py first.")
        sys.exit(1)
    print("\n[1/4] Loading DE gene lists...")
    top_genes_df = pd.read_csv(de_path)
    n_clusters = top_genes_df["cluster"].nunique()
    print(" ", len(top_genes_df), "gene-cluster pairs across", n_clusters, "clusters")
    adata_path = os.path.join(PROCESSED_DIR, "adata_de.h5ad")
    adata = sc.read_h5ad(adata_path) if os.path.exists(adata_path) else None
    print("\n[2/4] Submitting to Enrichr LINCS L1000...")
    print(" ", n_clusters, "clusters x up+down x", len(ENRICHR_LIBRARIES), "libraries")
    all_results = []
    clusters = top_genes_df["cluster"].unique()
    for i, cl in enumerate(clusters):
        print("  Cluster", cl, "(" + str(i+1) + "/" + str(len(clusters)) + ")...", end=" ", flush=True)
        up_genes = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "up"), "gene"].tolist()
        down_genes = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "down"), "gene"].tolist()
        cl_results = run_enrichr_for_cluster(cl, up_genes, down_genes)
        if not cl_results.empty:
            all_results.append(cl_results)
            n_hits = (cl_results["reversal_score"] > 0).sum()
            print(n_hits, "reversal hits")
        else:
            print("no results")
    if not all_results:
        print("ERROR: No Enrichr results returned.")
        sys.exit(1)
    raw_results = pd.concat(all_results, ignore_index=True)
    raw_path = os.path.join(PROCESSED_DIR, "lincs_results_raw.csv")
    raw_results.to_csv(raw_path, index=False)
    print("  Raw results:", len(raw_results), "rows ->", raw_path)
    pr_path = os.path.join(PROCESSED_DIR, "de_proraynaud_vs_rest.csv")
    if os.path.exists(pr_path):
        pr_df = pd.read_csv(pr_path)
        pr_up = pr_df[pr_df["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
        pr_down = pr_df[pr_df["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
        print("  Running focused pro-Raynaud query...")
        pr_results = run_enrichr_for_cluster("pro_raynaud_focused", pr_up, pr_down)
        if not pr_results.empty:
            pr_results.to_csv(os.path.join(PROCESSED_DIR, "lincs_proraynaud_focused.csv"), index=False)
    print("\n[3/4] Deduplicating and ranking...")
    candidates = deduplicate_and_rank(raw_results)
    if adata is not None and not candidates.empty and "cell_type" in adata.obs.columns:
        cluster_to_ct = adata.obs[["leiden", "cell_type"]].drop_duplicates().set_index("leiden")["cell_type"].to_dict()
        candidates["best_cluster_celltype"] = candidates["best_cluster"].apply(lambda x: cluster_to_ct.get(str(x), "unknown"))
    if not candidates.empty:
        print(" ", len(candidates), "unique compounds identified")
        print("\n  Top 10:")
        display_cols = ["compound", "max_reversal_score", "n_clusters"]
        if "best_cluster_celltype" in candidates.columns:
            display_cols.append("best_cluster_celltype")
        print(candidates[display_cols].head(10).round(2).to_string(index=False))
    print("\n[4/4] Saving...")
    cand_path = os.path.join(PROCESSED_DIR, "lincs_candidates.csv")
    candidates.to_csv(cand_path, index=False)
    print("\n" + "=" * 60)
    print("Script 06 complete.")
    print("  Raw LINCS results:", raw_path)
    print("  Ranked candidates:", cand_path, "(", len(candidates), "compounds )")
    print("=" * 60)

if __name__ == "__main__":
    main()
