"""Curation script to build an independent, multi-class protein/peptide evaluation dataset.

Fetches reviewed Swiss-Prot sequences with release dates, taxonomy, and ground truth
annotations for toxicity, allergenicity, and antigenicity from UniProt.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

QUERIES = [
    {
        "category": "toxin",
        "query": "reviewed:true AND keyword:KW-0800",
        "target_count": 100,
        "label_toxin": 1,
        "label_allergen": 0,
        "label_antigen": 0,
    },
    {
        "category": "toxin_short",
        "query": "reviewed:true AND keyword:KW-0800 AND length:[5 TO 14]",
        "target_count": 25,
        "label_toxin": 1,
        "label_allergen": 0,
        "label_antigen": 0,
    },
    {
        "category": "toxin_medium_short",
        "query": "reviewed:true AND keyword:KW-0800 AND length:[15 TO 50]",
        "target_count": 35,
        "label_toxin": 1,
        "label_allergen": 0,
        "label_antigen": 0,
    },
    {
        "category": "allergen",
        "query": "reviewed:true AND keyword:KW-0020 AND NOT keyword:KW-0800",
        "target_count": 100,
        "label_toxin": 0,
        "label_allergen": 1,
        "label_antigen": 0,
    },
    {
        "category": "antigen",
        "query": "reviewed:true AND (keyword:KW-0044 OR keyword:\"Surface antigen\") AND NOT keyword:KW-0800 AND NOT keyword:KW-0020",
        "target_count": 100,
        "label_toxin": 0,
        "label_allergen": 0,
        "label_antigen": 1,
    },
    {
        "category": "negative_control",
        "query": "reviewed:true AND (keyword:\"Ribosomal protein\" OR keyword:\"Metabolism\") AND NOT keyword:KW-0800 AND NOT keyword:KW-0020 AND NOT keyword:KW-0044",
        "target_count": 120,
        "label_toxin": 0,
        "label_allergen": 0,
        "label_antigen": 0,
    },
    {
        "category": "negative_short",
        "query": "reviewed:true AND length:[5 TO 14] AND NOT keyword:KW-0800 AND NOT keyword:KW-0020 AND NOT keyword:KW-0044",
        "target_count": 25,
        "label_toxin": 0,
        "label_allergen": 0,
        "label_antigen": 0,
    },
    {
        "category": "negative_medium_short",
        "query": "reviewed:true AND length:[15 TO 50] AND NOT keyword:KW-0800 AND NOT keyword:KW-0020 AND NOT keyword:KW-0044",
        "target_count": 35,
        "label_toxin": 0,
        "label_allergen": 0,
        "label_antigen": 0,
    },
]


def fetch_uniprot_entries(query: str, target_count: int) -> List[Dict[str, Any]]:
    """Fetch entries from UniProt REST API with pagination."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    entries = []
    size = min(50, target_count)
    params = {
        "query": query,
        "size": size,
    }

    req_url = url
    while req_url and len(entries) < target_count:
        try:
            resp = requests.get(req_url, params=params if req_url == url else None, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"UniProt query failed with status {resp.status_code}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                seq_obj = r.get("sequence", {})
                seq_str = seq_obj.get("value", "").strip().upper()
                if not seq_str:
                    continue
                # Ensure only standard amino acids
                if not all(aa in CANONICAL_AA for aa in seq_str):
                    continue

                acc = r.get("primaryAccession", "")
                name = r.get("uniProtkbId", acc)
                desc = r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
                org = r.get("organism", {}).get("scientificName", "")
                lineage = r.get("organism", {}).get("lineage", [])
                family = lineage[0] if lineage else "Unknown"
                audit = r.get("entryAudit", {})
                first_date = audit.get("firstPublicDate", "2010-01-01")

                entries.append({
                    "id": acc,
                    "entry_name": name,
                    "description": desc,
                    "organism": org,
                    "family": family,
                    "first_public_date": first_date,
                    "length": len(seq_str),
                    "sequence": seq_str,
                })
                if len(entries) >= target_count:
                    break

            # Check for next page in Link header
            link_header = resp.headers.get("Link", "")
            next_url = None
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip("<> ")
                        break
            req_url = next_url

        except Exception as e:
            logger.error(f"Error fetching from UniProt: {e}")
            break

    return entries


def main():
    logger.info("Curating independent evaluation dataset from Swiss-Prot...")
    dataset = []
    seen_ids = set()

    for q in QUERIES:
        cat = q["category"]
        logger.info(f"Fetching {cat} entries (query: {q['query']})...")
        records = fetch_uniprot_entries(q["query"], q["target_count"])
        logger.info(f"Retrieved {len(records)} entries for {cat}")

        for rec in records:
            if rec["id"] in seen_ids:
                continue
            seen_ids.add(rec["id"])
            rec["category"] = cat
            rec["is_toxic"] = q["label_toxin"]
            rec["is_allergen"] = q["label_allergen"]
            rec["is_antigen"] = q["label_antigen"]
            dataset.append(rec)

    # Sort deterministically
    dataset.sort(key=lambda x: x["id"])
    logger.info(f"Total curated dataset size: {len(dataset)} sequences")

    # Save JSON and FASTA
    out_json = Path("data/evaluation_dataset.json")
    out_fa = Path("data/evaluation_dataset.fa")

    out_json.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    with open(out_fa, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(f">{item['id']} {item['entry_name']} | {item['category']} | {item['first_public_date']} | len:{item['length']}\n{item['sequence']}\n")

    logger.info(f"Saved dataset to {out_json} and {out_fa}")


if __name__ == "__main__":
    main()
