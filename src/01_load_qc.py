import os, sys, glob
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import mygene

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW_H5_DIR = os.path.join(DATA_DIR, "raw", "GSE138669")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

QC_MIN_GENES = 200
QC_MIN_CELLS = 3
QC_MAX_MITO_PCT = 20.0
N_HVG = 4000
EC_MARKERS = ["VWF", "PECAM1", "CLDN5", "CDH5", "ENG", "PTPRB"]
PERICYTE_MARKERS = ["PDGFRB", "RGS5", "ACTA2", "NOTCH3", "DES", "CNN1"]
ALL_VASCULAR_MARKERS = EC_MARKERS + PERICYTE_MARKERS
VASCULAR_EXPR_THRESHOLD = 0.5
VASCULAR_MIN_MARKERS = 1
N_DEV_SAMPLES = 2

def find_h5_samples(h5_dir, n_samples=None):
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*_feature_bc_matrix.h5")))
    if not h5_files:
        print("ERROR: No .h5 files found in", h5_dir)
        sys.exit(1)
    samples = [(os.path.basename(f).replace("raw_feature_bc_matrix.h5","").rstrip("_"), f) for f in h5_files]
    if n_samples:
        samples = samples[:n_samples]
    print("  Found", len(samples), ".h5 files")
    return samples

def load_10x_h5(h5_path, sample_id):
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    adata.obs["sample"] = sample_id
    adata.obs_names = [sample_id + "_" + bc for bc in adata.obs_names]
    return adata

def map_gene_symbols(adata):
    mg = mygene.MyGeneInfo()
    gene_ids = adata.var_names.tolist()
    print("  Mapping", len(gene_ids), "gene IDs...")
    result = mg.querymany(gene_ids, scopes="ensembl.gene", fields="symbol", species="human", returnall=False, verbose=False)
    id_to_symbol = {r["query"]: r["symbol"] for r in result if "symbol" in r and "query" in r}
    new_names = [id_to_symbol.get(g, g) for g in gene_ids]
    seen = {}
    deduped = []
    for name in new_names:
        if name in seen:
            seen[name] += 1
            deduped.append(name + "_" + str(seen[name]))
        else:
            seen[name] = 0
            deduped.append(name)
    adata.var_names = deduped
    print("  Mapped", sum(1 for g in gene_ids if g in id_to_symbol), "/", len(gene_ids))
    return adata

def apply_qc(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=QC_MIN_CELLS)
    adata = adata[adata.obs["pct_counts_mt"] < QC_MAX_MITO_PCT].copy()
    print("  QC:", n_before, "->", adata.n_obs, "cells")
    return adata

def isolate_vascular_cells(adata):
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    present_markers = [m for m in ALL_VASCULAR_MARKERS if m in adata_norm.var_names]
    missing = [m for m in ALL_VASCULAR_MARKERS if m not in adata_norm.var_names]
    if missing:
        print("  Missing markers:", missing)
    print("  Using", len(present_markers), "markers:", present_markers)
    marker_expr = adata_norm[:, present_markers].X
    if hasattr(marker_expr, "toarray"):
        marker_expr = marker_expr.toarray()
    n_markers_expressed = (marker_expr > VASCULAR_EXPR_THRESHOLD).sum(axis=1)
    vascular_mask = n_markers_expressed >= VASCULAR_MIN_MARKERS
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
    cell_type = np.where(~vascular_mask, "non-vascular", np.where(ec_score >= pc_score, "endothelial", "pericyte"))
    adata.obs["vascular_subtype_prelim"] = cell_type
    adata_vasc = adata[vascular_mask].copy()
    n_ec = (adata_vasc.obs["vascular_subtype_prelim"] == "endothelial").sum()
    n_pc = (adata_vasc.obs["vascular_subtype_prelim"] == "pericyte").sum()
    print("  Vascular cells:", adata_vasc.n_obs, "(", n_ec, "EC,", n_pc, "pericyte)")
    print("  Excluded:", adata.n_obs - adata_vasc.n_obs)
    return adata_vasc

def select_hvg(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["norm_log"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", layer="counts", batch_key="sample", subset=False)
    print("  Selected", adata.var["highly_variable"].sum(), "HVGs")
    return adata

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 01: Load, QC, Vascular Filter")
    print("=" * 60)
    print("\n[1/6] Finding .h5 files...")
    samples = find_h5_samples(RAW_H5_DIR, n_samples=N_DEV_SAMPLES)
    print("\n[2/6] Loading .h5 files...")
    adatas = {}
    for sid, h5_path in samples:
        print("  Loading", sid, "...", end=" ", flush=True)
        try:
            adata = load_10x_h5(h5_path, sid)
            adatas[sid] = adata
            print(adata.n_obs, "cells x", adata.n_vars, "genes")
        except Exception as e:
            print("ERROR:", e)
    if not adatas:
        print("ERROR: No samples loaded.")
        sys.exit(1)
    print("\n  Concatenating", len(adatas), "samples...")
    adata_combined = sc.concat(list(adatas.values()), join="outer", fill_value=0)
    print("  Combined:", adata_combined.n_obs, "cells x", adata_combined.n_vars, "genes")
    print("\n[3/6] Mapping gene IDs...")
    adata_combined = map_gene_symbols(adata_combined)
    adata_combined.layers["counts"] = sp.csr_matrix(adata_combined.X) if not sp.issparse(adata_combined.X) else adata_combined.X.copy()
    print("\n[4/6] QC filtering...")
    adata_qc = apply_qc(adata_combined)
    qc_path = os.path.join(PROCESSED_DIR, "raynaud_qc.h5ad")
    adata_qc.write_h5ad(qc_path)
    pd.DataFrame([{"sample": sid, "n_cells": a.n_obs} for sid, a in adatas.items()]).to_csv(os.path.join(PROCESSED_DIR, "qc_summary.csv"), index=False)
    print("  Saved:", adata_qc.n_obs, "cells ->", qc_path)
    print("\n[5/6] Isolating vascular cells...")
    adata_vasc = isolate_vascular_cells(adata_qc)
    if adata_vasc.n_obs < 100:
        print("WARNING: Very few vascular cells. Consider lowering VASCULAR_EXPR_THRESHOLD.")
    print("\n[6/6] Selecting HVGs...")
    adata_vasc = select_hvg(adata_vasc)
    vasc_path = os.path.join(PROCESSED_DIR, "adata_vascular.h5ad")
    adata_vasc.write_h5ad(vasc_path)
    print("\n" + "=" * 60)
    print("Script 01 complete.")
    print("  Full QC:", adata_qc.n_obs, "cells ->", qc_path)
    print("  Vascular:", adata_vasc.n_obs, "cells ->", vasc_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
