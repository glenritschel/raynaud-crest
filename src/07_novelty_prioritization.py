import os, sys, time
import numpy as np
import pandas as pd
import requests

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_DELAY = 0.4
NCBI_EMAIL = "glen.ritschel@ritschelresearch.com"
NOVELTY_WEIGHTS = {"NOVEL_ALL": 3.0, "NOVEL_RAYNAUD": 1.5, "KNOWN": 1.0}
PATENT_WATCH_MIN_REVERSAL = 20.0
PATENT_WATCH_MIN_CLUSTERS = 2

MOA_REFERENCE = {
    "pd-0325901": "MEK1/2 inhibitor",
    "selumetinib": "MEK1/2 inhibitor",
    "trametinib": "MEK1/2 inhibitor",
    "pd-184352": "MEK1/2 inhibitor",
    "azd-8330": "MEK1/2 inhibitor",
    "ldn-193189": "BMP receptor ALK2/ALK3 inhibitor",
    "wye-125132": "mTORC1/2 inhibitor",
    "as-605240": "PI3K-gamma inhibitor",
    "tg-101348": "JAK2 inhibitor (fedratinib)",
    "fedratinib": "JAK2 inhibitor",
    "palbociclib": "CDK4/6 inhibitor",
    "radicicol": "HSP90 inhibitor",
    "geldanamycin": "HSP90 inhibitor",
    "withaferin-a": "NF-kB/HSP90 inhibitor",
    "celastrol": "NF-kB/HSP90 inhibitor",
    "chelerythrine": "PKC inhibitor",
    "chelerythrine chloride": "PKC inhibitor",
    "lapatinib": "EGFR/HER2 dual inhibitor",
    "canertinib": "Pan-EGFR inhibitor",
    "gefitinib": "EGFR inhibitor",
    "afatinib": "Pan-EGFR inhibitor",
    "pelitinib": "Pan-EGFR inhibitor",
    "cgp-60474": "CDK1/2 inhibitor",
    "bi-2536": "PLK1 inhibitor",
    "xmd-1150": "ERK5 inhibitor",
    "wz-3105": "SRC/ABL inhibitor",
    "wz-4-145": "CDK8 inhibitor",
    "saracatinib": "SRC/ALK2 dual inhibitor",
    "alvocidib": "CDK1/2/4/6/9 inhibitor (flavopiridol)",
    "fasudil": "ROCK inhibitor",
    "sildenafil": "PDE5 inhibitor",
    "imatinib": "BCR-ABL/PDGFR/KIT inhibitor",
    "dasatinib": "SRC/BCR-ABL inhibitor",
    "mitoxantrone": "Topoisomerase II inhibitor",
    "plx-4720": "BRAF V600E inhibitor",
    "ql-xii-47": "MELK/FLT3 inhibitor",
    "pi-103": "PI3K/mTOR dual inhibitor",
}

def pubmed_hit_count(query, retries=3):
    params = {"db": "pubmed", "term": query, "rettype": "count", "retmode": "json", "email": NCBI_EMAIL}
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
                return -1

def assess_novelty(compound_name):
    name_q = '"' + compound_name + '"'
    hits_raynaud = pubmed_hit_count(name_q + ' AND (Raynaud OR "Raynaud\'s phenomenon")')
    hits_ssc = pubmed_hit_count(name_q + ' AND ("systemic sclerosis" OR scleroderma)')
    hits_vasc = pubmed_hit_count(name_q + ' AND (vasospasm OR vasoconstriction OR "endothelial dysfunction")')
    if hits_raynaud == 0 and hits_ssc == 0 and hits_vasc == 0:
        tier = "NOVEL_ALL"
    elif hits_raynaud == 0 and hits_ssc == 0:
        tier = "NOVEL_RAYNAUD"
    else:
        tier = "KNOWN"
    return {"compound": compound_name, "hits_raynaud": hits_raynaud, "hits_ssc": hits_ssc, "hits_vasospasm": hits_vasc, "novelty_tier": tier}

def lookup_moa(compound_name):
    return MOA_REFERENCE.get(compound_name.lower().strip(), "unknown")

def main():
    print("=" * 60)
    print("RAYNAUD'S CREST PIPELINE - Script 07: Novelty & Priority Scoring")
    print("=" * 60)
    cand_path = os.path.join(PROCESSED_DIR, "lincs_candidates.csv")
    if not os.path.exists(cand_path):
        print("ERROR:", cand_path, "not found. Run 06_lincs_repurposing.py first.")
        sys.exit(1)
    print("\n[1/4] Loading LINCS candidates...")
    candidates = pd.read_csv(cand_path)
    print(" ", len(candidates), "candidates to assess")
    print("\n[2/4] Assessing novelty via PubMed...")
    print("  Querying", len(candidates), "compounds x 3 contexts...")
    novelty_rows = []
    for i, row in candidates.iterrows():
        compound = row["compound"]
        print("  [" + str(i+1) + "/" + str(len(candidates)) + "] " + compound + "...", end=" ", flush=True)
        novelty = assess_novelty(compound)
        novelty_rows.append(novelty)
        print(novelty["novelty_tier"], "(R:" + str(novelty["hits_raynaud"]) + ", SSc:" + str(novelty["hits_ssc"]) + ", V:" + str(novelty["hits_vasospasm"]) + ")")
    novelty_df = pd.DataFrame(novelty_rows)
    novelty_df.to_csv(os.path.join(PROCESSED_DIR, "novelty_raw.csv"), index=False)
    print("\n[3/4] Computing priority scores...")
    merged = candidates.merge(novelty_df, on="compound", how="left")
    merged["novelty_tier"] = merged["novelty_tier"].fillna("KNOWN")
    merged["moa"] = merged["compound"].apply(lookup_moa)
    merged["priority_score"] = merged.apply(
        lambda r: round(r["max_reversal_score"] * NOVELTY_WEIGHTS.get(r["novelty_tier"], 1.0) * r["n_clusters"], 1), axis=1
    )
    merged = merged.sort_values("priority_score", ascending=False)
    tier_counts = merged["novelty_tier"].value_counts()
    print("\n  Novelty breakdown:")
    for tier, count in tier_counts.items():
        print("    " + tier + ":", count, "compounds")
    display_cols = ["compound", "moa", "novelty_tier", "max_reversal_score", "n_clusters", "priority_score"]
    if "best_cluster_celltype" in merged.columns:
        display_cols.insert(2, "best_cluster_celltype")
    print("\n  Top 20 priority candidates:")
    print(merged[display_cols].head(20).round(2).to_string(index=False))
    patent_watch = merged[(merged["novelty_tier"] == "NOVEL_ALL") & (merged["max_reversal_score"] >= PATENT_WATCH_MIN_REVERSAL) & (merged["n_clusters"] >= PATENT_WATCH_MIN_CLUSTERS)].copy()
    print("\n  Patent watch list (NOVEL_ALL, reversal>=" + str(PATENT_WATCH_MIN_REVERSAL) + ", clusters>=" + str(PATENT_WATCH_MIN_CLUSTERS) + "):", len(patent_watch), "compounds")
    if not patent_watch.empty:
        print(patent_watch[display_cols].to_string(index=False))
    print("\n[4/4] Saving...")
    merged.to_csv(os.path.join(PROCESSED_DIR, "priority_candidates.csv"), index=False)
    patent_watch.to_csv(os.path.join(PROCESSED_DIR, "patent_watch.csv"), index=False)
    print("\n" + "=" * 60)
    print("Script 07 complete.")
    print("  Priority candidates:", len(merged), "->", os.path.join(PROCESSED_DIR, "priority_candidates.csv"))
    print("  Patent watch:", len(patent_watch), "->", os.path.join(PROCESSED_DIR, "patent_watch.csv"))
    print("=" * 60)
    print("\nPIPELINE COMPLETE. Review priority_candidates.csv for drug candidates.")

if __name__ == "__main__":
    main()
