"""
RAYNAUD'S CREST PIPELINE
Script 01: Load GSE138669, quality control, and vascular cell isolation

Key improvement over calcinosis pipeline:
- Adds marker-based vascular cell filtering BEFORE scVI embedding
- Isolates endothelial cells (ECs) and pericytes as a vascular subset
- Uses broad marker panel with low threshold to avoid losing transitional cells
- Saves both full QC'd object and vascular subset for downstream use

Outputs:
  data/adata_qc.h5ad        — full QC'd dataset (all cell types)
  data/adata_vascular.h5ad  — vascular subset (ECs + pericytes)
  data/qc_summary.csv       — per-sample QC stats
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import mygene
import requests
import gzip
import shutil
import tempfile

# ── Config ────────────────────────────────────────────────────────────────────
GEO_ACCESSION = "GSE138669"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# QC thresholds — matched to calcinosis pipeline
QC_MIN_GENES = 200
QC_MIN_CELLS = 3
QC_MAX_MITO_PCT = 20.0

# HVG selection — matched to calcinosis pipeline
N_HVG = 4000

# Vascular marker panel — broad to avoid losing transitional cells
# ECs: canonical pan-endothelial markers
EC_MARKERS = ["VWF", "PECAM1", "CLDN5", "CDH5", "ENG", "PTPRB"]
# Pericytes / vascular smooth muscle
PERICYTE_MARKERS = ["PDGFRB", "RGS5", "ACTA2", "NOTCH3", "DES", "CNN1"]
ALL_VASCULAR_MARKERS = EC_MARKERS + PERICYTE_MARKERS

# Minimum normalised expression to count a marker as "expressed" in a cell
VASCULAR_EXPR_THRESHOLD = 0.5

# Minimum number of vascular markers a cell must express to be included
VASCULAR_MIN_MARKERS = 1  # broad net — scVI will refine the clusters

# Dev mode: use only N_DEV_SAMPLES samples for fast iteration
# Set to None to use all 22 samples (full run on Colab)
N_DEV_SAMPLES = None  # None = full run


def download_geo_samples(accession, data_dir, n_samples=None):
    """
    Download 10x h5 files from GEO for GSE138669.
    Returns list of (sample_id, filepath) tuples.
    
    GSE138669 sample structure:
      Each sample is a directory containing barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz
    """
    import urllib.request

    # GEO FTP base for GSE138669
    # Sample IDs: GSM4118665 through GSM4118686 (22 samples)
    sample_ids = [f"GSM41186{65 + i}" for i in range(22)]
    if n_samples:
        sample_ids = sample_ids[:n_samples]

    downloaded = []
    for sid in sample_ids:
        sample_dir = os.path.join(data_dir, sid)
        os.makedirs(sample_dir, exist_ok=True)

        # Check if already downloaded
        mtx_path = os.path.join(sample_dir, "matrix.mtx.gz")
        if os.path.exists(mtx_path):
            print(f"  {sid}: already downloaded, skipping")
            downloaded.append((sid, sample_dir))
            continue

        # Download from GEO FTP
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sid[:7]}nnn/{sid}/suppl/"
        for fname in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
            url = base_url + f"{sid}_{fname}"
            dest = os.path.join(sample_dir, fname)
            print(f"  Downloading {sid}/{fname}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                # Try alternate naming convention
                url2 = base_url + fname
                try:
                    urllib.request.urlretrieve(url2, dest)
                except Exception as e2:
                    print(f"    WARNING: Could not download {fname}: {e2}")

        downloaded.append((sid, sample_dir))

    return downloaded


def load_10x_mtx(sample_dir, sample_id):
    """Load a 10x MTX directory into AnnData."""
    adata = sc.read_10x_mtx(
        sample_dir,
        var_names="gene_ids",
        cache=False
    )
    adata.obs["sample"] = sample_id
    return adata


def map_gene_symbols(adata):
    """
    Map Ensembl IDs to gene symbols using mygene.
    Genes that can't be mapped retain their Ensembl ID.
    Matched to calcinosis pipeline behaviour.
    """
    mg = mygene.MyGeneInfo()
    gene_ids = adata.var_names.tolist()
    
    print(f"  Mapping {len(gene_ids)} gene IDs to symbols...")
    result = mg.querymany(
        gene_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        returnall=False,
        verbose=False
    )
    
    id_to_symbol = {}
    for r in result:
        if "symbol" in r and "query" in r:
            id_to_symbol[r["query"]] = r["symbol"]
    
    new_var_names = [id_to_symbol.get(g, g) for g in gene_ids]
    
    # Handle duplicate symbols by appending suffix
    seen = {}
    deduped = []
    for name in new_var_names:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)
    
    adata.var_names = deduped
    mapped = sum(1 for g in gene_ids if g in id_to_symbol)
    print(f"  Mapped {mapped}/{len(gene_ids)} genes ({100*mapped/len(gene_ids):.1f}%)")
    return adata


def apply_qc(adata):
    """Apply QC filters matched to calcinosis pipeline."""
    # Mitochondrial gene detection
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    
    n_before = adata.n_obs
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=QC_MIN_CELLS)
    # Filter high mito
    adata = adata[adata.obs["pct_counts_mt"] < QC_MAX_MITO_PCT].copy()
    
    n_after = adata.n_obs
    print(f"  QC: {n_before} → {n_after} cells retained "
          f"({100*n_after/n_before:.1f}%)")
    return adata


def isolate_vascular_cells(adata):
    """
    Filter AnnData to vascular cells (ECs + pericytes) using marker expression.
    
    Strategy: use normalised (not raw) expression to score each cell
    for vascular marker co-expression. Cells expressing >= VASCULAR_MIN_MARKERS
    markers above VASCULAR_EXPR_THRESHOLD are retained.
    
    Improvement over calcinosis: explicit cell type isolation before scVI,
    preventing fibroblast signal from dominating the embedding.
    """
    # Normalise a temporary copy for marker scoring
    # (raw counts are preserved in adata.layers["counts"])
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # Find which vascular markers are present in the dataset
    present_markers = [m for m in ALL_VASCULAR_MARKERS if m in adata_norm.var_names]
    missing_markers = [m for m in ALL_VASCULAR_MARKERS if m not in adata_norm.var_names]
    
    if missing_markers:
        print(f"  Vascular markers not found in dataset: {missing_markers}")
    print(f"  Using {len(present_markers)} vascular markers: {present_markers}")
    
    # Score each cell: count how many markers exceed threshold
    marker_expr = adata_norm[:, present_markers].X
    if hasattr(marker_expr, "toarray"):
        marker_expr = marker_expr.toarray()
    
    n_markers_expressed = (marker_expr > VASCULAR_EXPR_THRESHOLD).sum(axis=1)
    vascular_mask = n_markers_expressed >= VASCULAR_MIN_MARKERS
    
    # Also annotate EC vs pericyte for each retained cell
    present_ec = [m for m in EC_MARKERS if m in adata_norm.var_names]
    present_pc = [m for m in PERICYTE_MARKERS if m in adata_norm.var_names]
    
    ec_expr = adata_norm[:, present_ec].X if present_ec else None
    pc_expr = adata_norm[:, present_pc].X if present_pc else None
    
    if ec_expr is not None and hasattr(ec_expr, "toarray"):
        ec_expr = ec_expr.toarray()
    if pc_expr is not None and hasattr(pc_expr, "toarray"):
        pc_expr = pc_expr.toarray()
    
    ec_score = ec_expr.mean(axis=1) if ec_expr is not None else np.zeros(adata.n_obs)
    pc_score = pc_expr.mean(axis=1) if pc_expr is not None else np.zeros(adata.n_obs)
    
    adata.obs["ec_score"] = ec_score
    adata.obs["pericyte_score"] = pc_score
    adata.obs["n_vascular_markers"] = n_markers_expressed
    adata.obs["putative_vascular"] = vascular_mask
    
    # Annotate predominant vascular subtype
    cell_type = np.where(
        ~vascular_mask, "non-vascular",
        np.where(ec_score >= pc_score, "endothelial", "pericyte")
    )
    adata.obs["vascular_subtype_prelim"] = cell_type
    
    # Subset to vascular cells
    adata_vasc = adata[vascular_mask].copy()
    
    n_ec = (adata_vasc.obs["vascular_subtype_prelim"] == "endothelial").sum()
    n_pc = (adata_vasc.obs["vascular_subtype_prelim"] == "pericyte").sum()
    print(f"  Vascular cells isolated: {adata_vasc.n_obs} total "
          f"({n_ec} putative EC, {n_pc} putative pericyte)")
    print(f"  Non-vascular cells excluded: {adata.n_obs - adata_vasc.n_obs}")
    
    return adata_vasc


def select_hvg(adata):
    """Select highly variable genes for scVI input."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Store normalised counts
    adata.layers["norm_log"] = adata.X.copy()
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_HVG,
        flavor="seurat_v3",
        layer="counts",
        batch_key="sample",
        subset=False  # keep all genes, just flag HVGs
    )
    
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  Selected {n_hvg} highly variable genes")
    return adata


def compute_qc_summary(adatas_by_sample):
    """Compute per-sample QC summary table."""
    rows = []
    for sid, adata in adatas_by_sample.items():
        rows.append({
            "sample": sid,
            "n_cells": adata.n_obs,
            "median_genes": adata.obs["n_genes_by_counts"].median(),
            "median_mito_pct": adata.obs["pct_counts_mt"].median(),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 01: Load, QC, Vascular Filter")
    print("=" * 60)

    # ── Step 1: Download ───────────────────────────────────────────────────
    print("\n[1/6] Downloading GSE138669 samples...")
    samples = download_geo_samples(GEO_ACCESSION, DATA_DIR, n_samples=N_DEV_SAMPLES)
    print(f"  {len(samples)} samples ready")

    # ── Step 2: Load and concatenate ──────────────────────────────────────
    print("\n[2/6] Loading 10x MTX files...")
    adatas = {}
    for sid, sample_dir in samples:
        print(f"  Loading {sid}...")
        try:
            adata = load_10x_mtx(sample_dir, sid)
            adatas[sid] = adata
            print(f"    {adata.n_obs} cells × {adata.n_vars} genes")
        except Exception as e:
            print(f"    ERROR loading {sid}: {e}")

    if not adatas:
        print("ERROR: No samples loaded. Exiting.")
        sys.exit(1)

    print(f"\n  Concatenating {len(adatas)} samples...")
    adata_combined = sc.concat(
        list(adatas.values()),
        label="sample",
        keys=list(adatas.keys()),
        join="outer",
        fill_value=0
    )
    print(f"  Combined: {adata_combined.n_obs} cells × {adata_combined.n_vars} genes")

    # ── Step 3: Gene symbol mapping ───────────────────────────────────────
    print("\n[3/6] Mapping gene IDs to symbols...")
    adata_combined = map_gene_symbols(adata_combined)

    # Store raw counts before any normalisation
    import scipy.sparse as sp
    adata_combined.layers["counts"] = (
        sp.csr_matrix(adata_combined.X)
        if not sp.issparse(adata_combined.X)
        else adata_combined.X.copy()
    )

    # ── Step 4: QC filtering ──────────────────────────────────────────────
    print("\n[4/6] Applying QC filters...")
    # Per-sample QC stats before filtering
    qc_before = compute_qc_summary(adatas)
    
    adata_qc = apply_qc(adata_combined)

    # Save full QC'd object
    qc_path = os.path.join(DATA_DIR, "adata_qc.h5ad")
    print(f"\n  Saving full QC object to {qc_path}...")
    adata_qc.write_h5ad(qc_path)
    qc_before.to_csv(os.path.join(DATA_DIR, "qc_summary.csv"), index=False)
    print(f"  Saved: {adata_qc.n_obs} cells × {adata_qc.n_vars} genes")

    # ── Step 5: Vascular cell isolation ───────────────────────────────────
    print("\n[5/6] Isolating vascular cells (ECs + pericytes)...")
    adata_vasc = isolate_vascular_cells(adata_qc)

    if adata_vasc.n_obs < 100:
        print("WARNING: Very few vascular cells found. "
              "Consider lowering VASCULAR_EXPR_THRESHOLD or VASCULAR_MIN_MARKERS.")

    # ── Step 6: HVG selection on vascular subset ──────────────────────────
    print("\n[6/6] Selecting highly variable genes on vascular subset...")
    adata_vasc = select_hvg(adata_vasc)

    vasc_path = os.path.join(DATA_DIR, "adata_vascular.h5ad")
    print(f"\n  Saving vascular subset to {vasc_path}...")
    adata_vasc.write_h5ad(vasc_path)

    print("\n" + "=" * 60)
    print("Script 01 complete.")
    print(f"  Full QC dataset:    {adata_qc.n_obs} cells  → {qc_path}")
    print(f"  Vascular subset:    {adata_vasc.n_obs} cells → {vasc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
