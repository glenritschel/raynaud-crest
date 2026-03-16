"""
RAYNAUD'S CREST PIPELINE
Script 07: Novelty assessment and priority scoring

Matched to calcinosis pipeline:
- PubMed hit counts via NCBI E-utilities for each compound
- Three query contexts: Raynaud's, SSc/scleroderma, vasospasm/vasoconstriction
- Novelty tiers: NOVEL_ALL, NOVEL_RAYNAUD, KNOWN
- Priority score = max_reversal_score × novelty_weight × n_clusters

Improvement over calcinosis:
- Query contexts updated for Raynaud's biology
  (calcinosis queried: calcinosis, SSc, calcification)
  (Raynaud's queries: Raynaud's, SSc, vasospasm)
- Adds MOA lookup from a curated reference table
- Outputs a patent-watch list (NOVEL_ALL candidates with strong reversal scores)

Inputs:  data/lincs_candidates.csv
Outputs: data/priority_candidates.csv   — final ranked table
         data/patent_watch.csv          — NOVEL_ALL high-priority candidates
         data/novelty_raw.csv           — raw PubMed hit counts
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_DELAY = 0.4   # NCBI rate limit: 3 req/sec without API key
NCBI_EMAIL = "glen.ritschel@ritschelresearch.com"

# Novelty weights — matched to calcinosis pipeline
NOVELTY_WEIGHTS = {
    "NOVEL_ALL":     3.0,
    "NOVEL_RAYNAUD": 1.5,
    "KNOWN":         1.0,
}

# Patent watch threshold
PATENT_WATCH_MIN_REVERSAL = 20.0
PATENT_WATCH_MIN_CLUSTERS = 2

# Curated MOA reference — extended from calcinosis pipeline
# Maps compound name (lowercase) to mechanism of action
MOA_REFERENCE = {
    # Kinase inhibitors
    "pd-0325901":     "MEK1/2 inhibitor",
    "ld n-193189":    "BMP receptor ALK2/ALK3 inhibitor",
    "ldn-193189":     "BMP receptor ALK2/ALK3 inhibitor",
    "wye-125132":     "mTORC1/2 inhibitor",
    "as-605240":      "PI3K-gamma inhibitor",
    "tg-101348":      "JAK2 inhibitor (fedratinib)",
    "fedratinib":     "JAK2 inhibitor",
    "palbociclib":    "CDK4/6 inhibitor",
    "radicicol":      "HSP90 inhibitor",
    "geldanamycin":   "HSP90 inhibitor",
    "pi-103":         "PI3K/mTOR dual inhibitor",
    "bi-2536":        "PLK1 inhibitor",
    "foretinib":      "MET/VEGFR2 inhibitor",
    "wz-3105":        "SRC/ABL inhibitor",
    "cgp-60474":      "CDK1/2 inhibitor",
    # Vasodilators / vascular targets
    "sildenafil":     "PDE5 inhibitor",
    "tadalafil":      "PDE5 inhibitor",
    "bosentan":       "Endothelin receptor antagonist",
    "macitentan":     "Endothelin receptor antagonist",
    "ambrisentan":    "Endothelin receptor-A antagonist",
    "iloprost":       "Prostacyclin analogue",
    "treprostinil":   "Prostacyclin analogue",
    "nifedipine":     "Calcium channel blocker",
    "amlodipine":     "Calcium channel blocker",
    "fasudil":        "ROCK inhibitor",
    "y-27632":        "ROCK inhibitor",
    # Anti-fibrotic
    "nintedanib":     "FGFR/PDGFR/VEGFR inhibitor",
    "imatinib":       "BCR-ABL/PDGFR/KIT inhibitor",
    "dasatinib":      "SRC/BCR-ABL inhibitor",
    "tocilizumab":    "IL-6 receptor antibody",
    "tofacitinib":    "JAK1/3 inhibitor",
    "baricitinib":    "JAK1/2 inhibitor",
    # Natural compounds
    "withaferin-a":   "NF-kB/HSP90 inhibitor",
    "celastrol":      "NF-kB/HSP90 inhibitor",
    "chelerythrine":  "PKC inhibitor",
}


def pubmed_hit_count(query, retries=3):
    """Query PubMed and return hit count for a search string."""
    params = {
        "db": "pubmed",
        "term": query,
        "rettype": "count",
        "retmode": "json",
        "email": NCBI_EMAIL,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(NCBI_ESEARCH, params=params, timeout=10)
            resp.raise_for_status()
            count = int(resp.json()["esearchresult"]["count"])
            time.sleep(NCBI_DELAY)
            return count
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(NCBI_DELAY * 3)
            else:
                print(f"    WARNING: PubMed query failed: {e}")
                return -1


def assess_novelty(compound_name):
    """
    Query PubMed for compound in three Raynaud's-relevant contexts.
    Returns dict with hit counts and novelty tier.
    """
    name_q = f'"{compound_name}"'

    # Context 1: Raynaud's phenomenon specifically
    q_raynaud = f"{name_q} AND (Raynaud OR \"Raynaud's phenomenon\")"
    # Context 2: SSc / scleroderma broadly
    q_ssc = f"{name_q} AND (\"systemic sclerosis\" OR scleroderma)"
    # Context 3: vasospasm / vasoconstriction (mechanism class)
    q_vasc = f"{name_q} AND (vasospasm OR vasoconstriction OR \"endothelial dysfunction\")"

    hits_raynaud = pubmed_hit_count(q_raynaud)
    hits_ssc = pubmed_hit_count(q_ssc)
    hits_vasc = pubmed_hit_count(q_vasc)

    # Classify novelty
    if hits_raynaud == 0 and hits_ssc == 0 and hits_vasc == 0:
        tier = "NOVEL_ALL"
    elif hits_raynaud == 0 and hits_ssc == 0:
        tier = "NOVEL_RAYNAUD"
    else:
        tier = "KNOWN"

    return {
        "compound": compound_name,
        "hits_raynaud": hits_raynaud,
        "hits_ssc": hits_ssc,
        "hits_vasospasm": hits_vasc,
        "novelty_tier": tier,
    }


def compute_priority_score(row):
    """
    Priority score = max_reversal_score × novelty_weight × n_clusters
    Matched to calcinosis pipeline formula.
    """
    weight = NOVELTY_WEIGHTS.get(row["novelty_tier"], 1.0)
    return round(row["max_reversal_score"] * weight * row["n_clusters"], 1)


def lookup_moa(compound_name):
    """Look up MOA from reference table."""
    return MOA_REFERENCE.get(compound_name.lower().strip(), "unknown")


def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE — Script 07: Novelty & Priority Scoring")
    print("=" * 60)

    # ── Load candidates ───────────────────────────────────────────────────
    cand_path = os.path.join(DATA_DIR, "lincs_candidates.csv")
    if not os.path.exists(cand_path):
        print(f"ERROR: {cand_path} not found. Run 06_lincs_repurposing.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading LINCS candidates from {cand_path}...")
    candidates = pd.read_csv(cand_path)
    print(f"  {len(candidates)} candidates to assess")

    # ── Novelty assessment ────────────────────────────────────────────────
    print(f"\n[2/4] Assessing novelty via PubMed E-utilities...")
    print(f"  Querying {len(candidates)} compounds × 3 contexts "
          f"(~{len(candidates) * 3 * NCBI_DELAY:.0f}s estimated)...")

    novelty_rows = []
    for i, row in candidates.iterrows():
        compound = row["compound"]
        print(f"  [{i+1}/{len(candidates)}] {compound}...", end=" ", flush=True)
        novelty = assess_novelty(compound)
        novelty_rows.append(novelty)
        print(f"{novelty['novelty_tier']} "
              f"(R:{novelty['hits_raynaud']}, "
              f"SSc:{novelty['hits_ssc']}, "
              f"V:{novelty['hits_vasospasm']})")

    novelty_df = pd.DataFrame(novelty_rows)
    novelty_path = os.path.join(DATA_DIR, "novelty_raw.csv")
    novelty_df.to_csv(novelty_path, index=False)

    # ── Merge and score ───────────────────────────────────────────────────
    print(f"\n[3/4] Computing priority scores...")
    merged = candidates.merge(novelty_df, on="compound", how="left")
    merged["novelty_tier"] = merged["novelty_tier"].fillna("KNOWN")
    merged["moa"] = merged["compound"].apply(lookup_moa)
    merged["priority_score"] = merged.apply(compute_priority_score, axis=1)
    merged = merged.sort_values("priority_score", ascending=False)

    # Novelty summary
    tier_counts = merged["novelty_tier"].value_counts()
    print(f"\n  Novelty breakdown:")
    for tier, count in tier_counts.items():
        print(f"    {tier}: {count} compounds")

    print(f"\n  Top 20 priority candidates:")
    display_cols = ["compound", "moa", "novelty_tier", "max_reversal_score",
                    "n_clusters", "priority_score"]
    if "best_cluster_celltype" in merged.columns:
        display_cols.insert(2, "best_cluster_celltype")
    print(merged[display_cols].head(20).round(2).to_string(index=False))

    # ── Patent watch list ─────────────────────────────────────────────────
    patent_watch = merged[
        (merged["novelty_tier"] == "NOVEL_ALL") &
        (merged["max_reversal_score"] >= PATENT_WATCH_MIN_REVERSAL) &
        (merged["n_clusters"] >= PATENT_WATCH_MIN_CLUSTERS)
    ].copy()

    print(f"\n  Patent watch list (NOVEL_ALL, reversal≥{PATENT_WATCH_MIN_REVERSAL}, "
          f"clusters≥{PATENT_WATCH_MIN_CLUSTERS}): {len(patent_watch)} compounds")
    if not patent_watch.empty:
        print(patent_watch[display_cols].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[4/4] Saving final outputs...")
    priority_path = os.path.join(DATA_DIR, "priority_candidates.csv")
    patent_path = os.path.join(DATA_DIR, "patent_watch.csv")

    merged.to_csv(priority_path, index=False)
    patent_watch.to_csv(patent_path, index=False)

    print("\n" + "=" * 60)
    print("Script 07 complete.")
    print(f"  Priority candidates: {priority_path} ({len(merged)} compounds)")
    print(f"  Patent watch list:   {patent_path} ({len(patent_watch)} compounds)")
    print(f"  Novelty raw:         {novelty_path}")
    print("=" * 60)
    print("\nPIPELINE COMPLETE. Review priority_candidates.csv for drug candidates.")


if __name__ == "__main__":
    main()
