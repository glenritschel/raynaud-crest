"""
RAYNAUD'S CREST PIPELINE
Script 06: LINCS L1000 drug reversal scoring via Enrichr

Matched to calcinosis pipeline:
- Submits top 150 up + 150 down genes per cluster to Enrichr
- Queries LINCS_L1000_Chem_Pert_up and LINCS_L1000_Chem_Pert_down
- Computes signed reversal scores: -log10(adj_p), positive = reversal

Improvement over calcinosis:
- Also submits pro-Raynaud's vs rest DE gene list as an additional query
- Annotates each candidate with the EC subtype of its top-scoring cluster

Inputs:  data/adata_de.h5ad
         data/de_top_genes.csv
         data/de_proraynaud_vs_rest.csv  (optional)
Outputs: data/lincs_results_raw.csv
         data/lincs_candidates.csv       — deduplicated, ranked candidates
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import gseapy as gp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Matched to calcinosis pipeline
N_TOP_GENES = 150
TOP_PER_CLUSTER = 15   # top compounds per cluster before deduplication
ENRICHR_DELAY = 1.0    # seconds between API calls — be polite

ENRICHR_LIBRARIES = [
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Chem_Pert_down",
    # Additional pathway libraries for mechanistic context
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]


def clean_compound_name(term):
    """
    Extract compound name from Enrichr LINCS term string.
    LINCS terms are formatted as: 'CompoundName_CellLine_Dose_Time'
    """
    parts = term.split("_")
    return parts[0].strip() if parts else term.strip()


def run_enrichr_for_cluster(cluster_id, up_genes, down_genes):
    """
    Submit up and down gene lists for one cluster to Enrichr.
    Returns DataFrame of LINCS reversal candidates.
    """
    results = []

    for direction, genes, reversal_lib, same_lib in [
        ("up",   up_genes,   "LINCS_L1000_Chem_Pert_down", "LINCS_L1000_Chem_Pert_up"),
        ("down", down_genes, "LINCS_L1000_Chem_Pert_up",   "LINCS_L1000_Chem_Pert_down"),
    ]:
        if not genes:
            continue

        for lib in ENRICHR_LIBRARIES:
            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=lib,
                    outdir=None,
                    verbose=False,
                )
                df = enr.results.copy()
                if df.empty:
                    continue

                df["cluster"] = cluster_id
                df["query_direction"] = direction
                df["library"] = lib

                # Compute reversal score for LINCS libraries
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
                print(f"    WARNING: Enrichr query failed for cluster {cluster_id}, "
                      f"lib={lib}, dir={direction}: {e}")
                time.sleep(ENRICHR_DELAY * 2)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_all_clusters(top_genes_df):
    """
    Run Enrichr for all clusters. Returns combined raw results.
    """
    clusters = top_genes_df["cluster"].unique()
    all_results = []

    for i, cl in enumerate(clusters):
        print(f"  Cluster {cl} ({i+1}/{len(clusters)})...", end=" ", flush=True)

        up_genes = top_genes_df.loc[
            (top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "up"),
            "gene"
        ].tolist()

        down_genes = top_genes_df.loc[
            (top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "down"),
            "gene"
        ].tolist()

        cl_results = run_enrichr_for_cluster(cl, up_genes, down_genes)
        if not cl_results.empty:
            all_results.append(cl_results)
            n_lincs = (cl_results["reversal_score"] > 0).sum()
            print(f"{n_lincs} reversal hits")
        else:
            print("no results")

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def deduplicate_and_rank(raw_df, top_per_cluster=TOP_PER_CLUSTER):
    """
    Deduplicate LINCS results:
    1. Keep top N candidates per cluster by reversal score
    2. Deduplicate by compound name
    3. Aggregate: max reversal score, number of clusters, list of clusters

    Matched to calcinosis pipeline logic.
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Filter to LINCS libraries only and positive reversal scores
    lincs_mask = raw_df["library"].str.startswith("LINCS_L1000")
    lincs_df = raw_df[lincs_mask & (raw_df["reversal_score"] > 0)].copy()

    if lincs_df.empty:
        print("  WARNING: No positive LINCS reversal scores found.")
        return pd.DataFrame()

    # Top N per cluster
    top_per_cl = (
        lincs_df
        .sort_values("reversal_score", ascending=False)
        .groupby("cluster")
        .head(top_per_cluster)
    )

    # Deduplicate by compound
    compound_agg = (
        top_per_cl
        .groupby("compound")
        .agg(
            max_reversal_score=("reversal_score", "max"),
            n_clusters=("cluster", "nunique"),
            clusters=("cluster", lambda x: sorted(set(x.astype(str)))),
            best_cluster=("cluster", lambda x: x.loc[x.index[
                top_per_cl.loc[x.index, "reversal_score"].argmax()
            ]]),
        )
        .reset_index()
    )

    compound_agg["clusters"] = compound_agg["clusters"].apply(
        lambda x: ",".join(x)
    )

    return compound_agg.sort_values("max_reversal_score", ascending=False)


def annotate_best_cluster_celltype(candidates_df, adata):
    """
    Add the EC subtype annotation of each compound's best-scoring cluster.
    Improvement over calcinosis: adds biological context to each candidate.
    """
    if "cell_type" not in adata.obs.columns:
        return candidates_df

    cluster_to_celltype = (
        adata.obs[["leiden", "cell_type"]]
        .drop_duplicates()
        .set_index("leiden")["cell_type"]
        .to_dict()
    )

    candidates_df["best_cluster_celltype"] = candidates_df["best_cluster"].apply(
        lambda x: cluster_to_celltype.get(str(x), "unknown")
    )
    return candidates_df


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 06: LINCS L1000 Reversal Scoring")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    de_path = os.path.join(DATA_DIR, "de_top_genes.csv")
    if not os.path.exists(de_path):
        print(f"ERROR: {de_path} not found. Run 05_differential_expression.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading DE gene lists from {de_path}...")
    top_genes_df = pd.read_csv(de_path)
    n_clusters = top_genes_df["cluster"].nunique()
    print(f"  {len(top_genes_df):,} gene-cluster pairs across {n_clusters} clusters")

    # Load adata for cell type annotation
    adata_path = os.path.join(DATA_DIR, "adata_de.h5ad")
    import scanpy as sc
    adata = sc.read_h5ad(adata_path) if os.path.exists(adata_path) else None

    # ── Run Enrichr for all clusters ──────────────────────────────────────
    print(f"\n[2/4] Submitting gene lists to Enrichr LINCS L1000...")
    print(f"  {n_clusters} clusters × up+down × {len(ENRICHR_LIBRARIES)} libraries")
    print(f"  (This is the slow step — ~{ENRICHR_DELAY}s per API call)")

    raw_results = run_all_clusters(top_genes_df)

    if raw_results.empty:
        print("ERROR: No Enrichr results returned. Check network connectivity.")
        sys.exit(1)

    raw_path = os.path.join(DATA_DIR, "lincs_results_raw.csv")
    raw_results.to_csv(raw_path, index=False)
    print(f"  Raw results: {len(raw_results):,} rows → {raw_path}")

    # ── Also run pro-Raynaud's focused query ──────────────────────────────
    pr_path = os.path.join(DATA_DIR, "de_proraynaud_vs_rest.csv")
    if os.path.exists(pr_path):
        print(f"\n  Running focused query for pro-Raynaud's vs rest DE list...")
        pr_df = pd.read_csv(pr_path)
        pr_up = pr_df[pr_df["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
        pr_down = pr_df[pr_df["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
        pr_results = run_enrichr_for_cluster("pro_raynaud_focused", pr_up, pr_down)
        if not pr_results.empty:
            pr_results_path = os.path.join(DATA_DIR, "lincs_proraynaud_focused.csv")
            pr_results.to_csv(pr_results_path, index=False)
            print(f"  Focused results → {pr_results_path}")

    # ── Deduplicate and rank ──────────────────────────────────────────────
    print(f"\n[3/4] Deduplicating and ranking candidates...")
    candidates = deduplicate_and_rank(raw_results)

    if candidates.empty:
        print("WARNING: No candidates after deduplication.")
    else:
        if adata is not None:
            candidates = annotate_best_cluster_celltype(candidates, adata)

        print(f"  {len(candidates)} unique compounds identified")
        print(f"\n  Top 10 by reversal score:")
        display_cols = ["compound", "max_reversal_score", "n_clusters",
                        "best_cluster_celltype"] if "best_cluster_celltype" in candidates.columns \
                        else ["compound", "max_reversal_score", "n_clusters"]
        print(candidates[display_cols].head(10).round(2).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[4/4] Saving candidates...")
    cand_path = os.path.join(DATA_DIR, "lincs_candidates.csv")
    candidates.to_csv(cand_path, index=False)

    print("\n" + "=" * 60)
    print("Script 06 complete.")
    print(f"  Raw LINCS results:   {raw_path}")
    print(f"  Ranked candidates:   {cand_path}")
    print(f"  Total candidates:    {len(candidates)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
