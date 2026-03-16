# raynaud-crest

**Transcriptome-Guided Drug Repurposing for Raynaud's Phenomenon in Systemic Sclerosis**

Glen Ritschel | Ritschel Research | 2026

Part of the **CREST Implication Testing Series** — a systematic computational drug repurposing pipeline applied to each of the five CREST features of systemic sclerosis (SSc).

| Feature | Repo | Status |
|---|---|---|
| Calcinosis | [calcinosis-crest](https://github.com/glenritschel/calcinosis-crest) | Complete |
| Raynaud's | [raynaud-crest](https://github.com/glenritschel/raynaud-crest) | In progress |
| Esophageal dysmotility | — | Planned |
| Sclerodactyly | — | Planned |
| Telangiectasia | — | Planned |

---

## Overview

This pipeline applies scRNA-seq-guided drug repurposing to Raynaud's phenomenon in SSc, focusing on the vascular compartment of SSc skin. Key improvements over the calcinosis pipeline:

- **Marker-based vascular cell isolation** before scVI embedding, isolating endothelial cells (ECs) and pericytes from the fibroblast-dominated GSE138669 dataset
- **EC subtype annotation** (arterial, capillary, venous, lymphatic, pericyte) using canonical markers from Huang et al. 2024
- **Multi-resolution Leiden clustering** (0.5, 0.8, 1.2) with automated selection based on EC subtype marker separation
- **Raynaud's-specific gene signatures**: vasospasm, endothelial injury, impaired angiogenesis, oxidative stress, pericyte dysfunction
- **Focused pro-Raynaud's DE**: pro-Raynaud's clusters vs rest, in addition to cluster-vs-rest analysis

## Dataset

**GSE138669** — Tabib et al. 2021 (*Nature Communications*): 22 SSc skin biopsies, 57,156 cells after QC. All SSc; no healthy controls.

Raw data: [NCBI GEO GSE138669](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138669)

## Pipeline

```
01_load_qc.py          Load GSE138669, QC, vascular cell isolation (ECs + pericytes)
02_scvi_embed.py       scVI embedding + multi-resolution Leiden (GPU)
03_annotate_clusters.py  EC subtype annotation
04_signature_scoring.py  Raynaud's gene signature scoring
05_differential_expression.py  Wilcoxon DE per cluster
06_lincs_repurposing.py  LINCS L1000 reversal scoring via Enrichr
07_novelty_prioritization.py  PubMed novelty check + priority scoring
```

## Usage

### Colab (recommended — GPU required for scVI)

1. Open `notebooks/raynaud_pipeline.ipynb` in Google Colab
2. Set runtime to **T4 GPU**: Runtime > Change runtime type > T4 GPU
3. Set `N_DEV_SAMPLES = 2` for a fast test run, then `None` for the full 22-sample run
4. Run all cells in sequence

### Local (CPU only — slow scVI)

```bash
git clone https://github.com/glenritschel/raynaud-crest
cd raynaud-crest
conda env create -f environment.yml
conda activate raynaud-crest
python src/00_install.py
python src/01_load_qc.py
python src/02_scvi_embed.py   # slow on CPU
python src/03_annotate_clusters.py
python src/04_signature_scoring.py
python src/05_differential_expression.py
python src/06_lincs_repurposing.py
python src/07_novelty_prioritization.py
```

## Key outputs

| File | Description |
|---|---|
| `results/tables/de_leiden_wilcoxon.csv` | Wilcoxon DE results, all clusters |
| `results/lincs_reversal_top15_by_cluster.csv` | Top 15 LINCS candidates per cluster |
| `results/drug_repurposing/` | Per-cluster Enrichr reports |
| `data/processed/priority_candidates.csv` | Final ranked drug candidates |
| `data/processed/patent_watch.csv` | NOVEL_ALL high-priority candidates |
| `data/processed/cluster_annotations.csv` | EC subtype annotations per cluster |
| `data/processed/signature_scores.csv` | Raynaud's signature scores per cluster |
| `figures/` | UMAP plots |

## Repository structure

```
raynaud-crest/
├── README.md
├── environment.yml
├── notebooks/
│   └── raynaud_pipeline.ipynb     # Colab entry point
├── src/
│   ├── 00_install.py
│   ├── 01_load_qc.py
│   ├── 02_scvi_embed.py
│   ├── 03_annotate_clusters.py
│   ├── 04_signature_scoring.py
│   ├── 05_differential_expression.py
│   ├── 06_lincs_repurposing.py
│   └── 07_novelty_prioritization.py
├── data/
│   ├── raw/GSE138669/             # downloaded .h5 files (git-ignored)
│   └── processed/                 # h5ad outputs (git-ignored)
├── results/
│   ├── drug_repurposing/          # per-cluster Enrichr reports
│   ├── tables/
│   └── lincs_reversal_top15_by_cluster.csv
├── figures/
└── logs/
```

## References

- Tabib T et al. (2021). Myofibroblast transcriptome indicates SFRP2hi fibroblast progenitors in systemic sclerosis skin. *Nat Commun*, 12, 4384. [GSE138669]
- Huang M et al. (2024). Single-cell transcriptomes and chromatin accessibility of endothelial cells unravel transcription factors associated with dysregulated angiogenesis in systemic sclerosis. *Ann Rheum Dis*.
- Subramanian A et al. (2017). A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. *Cell*, 171(6), 1437–1452.
- Lopez R et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nat Methods*, 15(12), 1053–1058.
- Ritschel G, Claude (Anthropic). (2026). Transcriptome-Guided Drug Repurposing for Calcinosis in Systemic Sclerosis. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## License

CC BY 4.0 — Glen Ritschel, Ritschel Research, 2026
