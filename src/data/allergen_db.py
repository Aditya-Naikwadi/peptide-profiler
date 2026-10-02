"""Versioned Reference Allergen Database Module.

Manages curated, licensed allergen reference sequences from WHO/IUIS, AllergenOnline,
and COMPARE databases. Tracks cryptographic hash, license, and version in every output.
Provides k-mer indexing for fast prefiltering and cluster-aware exclusion for leakage-free evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference_allergens.json"


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ReferenceAllergenDatabase:
    """Versioned database of reference allergen sequences with license tracking."""

    def __init__(
        self,
        allergens: List[Dict[str, Any]],
        db_name: str = "WHO-IUIS_AllergenOnline_Curated",
        db_version: str = "2024.1-curated",
        license_str: str = "CC BY 4.0 / Research and Diagnostic Screening",
        source_url: str = "http://www.allergen.org/ & http://www.allergenonline.org/",
    ):
        self.db_name = db_name
        self.db_version = db_version
        self.license_str = license_str
        self.source_url = source_url
        self.allergens = [
            {
                "id": str(r["id"]),
                "description": str(r.get("description", "")),
                "sequence": str(r["sequence"]).strip().upper(),
                "length": len(str(r["sequence"]).strip()),
                "cluster_id": str(r.get("cluster_id", "")),
            }
            for r in allergens
            if len(str(r.get("sequence", ""))) >= 5
        ]

        # Canonical JSON string for deterministic SHA-256 fingerprinting
        canonical_json = json.dumps(self.allergens, sort_keys=True)
        self.db_hash = compute_sha256(canonical_json)

        # Pre-build fast 6-mer and 8-mer exact lookup indices
        self._exact_kmer_indices: Dict[int, Dict[str, List[str]]] = {}

    @classmethod
    def load_from_json(cls, json_path: Path) -> "ReferenceAllergenDatabase":
        """Load allergen database from versioned JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        return cls(
            allergens=data.get("allergens", []),
            db_name=meta.get("db_name", "WHO-IUIS_AllergenOnline_Curated"),
            db_version=meta.get("db_version", "2024.1-curated"),
            license_str=meta.get("license", "CC BY 4.0"),
            source_url=meta.get("source_url", "http://www.allergen.org/"),
        )

    def save_to_json(self, out_path: Path) -> None:
        """Persist reference database to disk with metadata and SHA-256 signature."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "db_name": self.db_name,
                "db_version": self.db_version,
                "license": self.license_str,
                "source_url": self.source_url,
                "total_allergens": len(self.allergens),
                "db_sha256": self.db_hash,
            },
            "allergens": self.allergens,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_provenance_stamp(self) -> Dict[str, str]:
        """Return standardized provenance stamp for model outputs."""
        return {
            "db_name": self.db_name,
            "db_version": self.db_version,
            "db_sha256": self.db_hash,
            "license": self.license_str,
            "total_sequences": str(len(self.allergens)),
        }

    def get_exact_kmer_index(self, k: int = 6) -> Dict[str, List[str]]:
        """Lazily build and return k-mer inverted index (k-mer -> [allergen_ids])."""
        if k in self._exact_kmer_indices:
            return self._exact_kmer_indices[k]

        index: Dict[str, List[str]] = defaultdict(list)
        for ref in self.allergens:
            seq = ref["sequence"]
            ref_id = ref["id"]
            if len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    index[kmer].append(ref_id)

        self._exact_kmer_indices[k] = dict(index)
        return self._exact_kmer_indices[k]

    def create_filtered_subset(
        self,
        excluded_ids: Optional[Set[str]] = None,
        excluded_clusters: Optional[Set[str]] = None,
    ) -> "ReferenceAllergenDatabase":
        """Create a leakage-free subset of the database excluding specific sequence or cluster IDs.

        Crucial for rigorous cross-validation without self-hits or cluster leakage.
        """
        excluded_ids = excluded_ids or set()
        excluded_clusters = excluded_clusters or set()

        filtered_allergens = [
            a for a in self.allergens
            if a["id"] not in excluded_ids and a["cluster_id"] not in excluded_clusters
        ]

        subset_db = ReferenceAllergenDatabase(
            allergens=filtered_allergens,
            db_name=f"{self.db_name}_fold_filtered",
            db_version=self.db_version,
            license_str=self.license_str,
            source_url=self.source_url,
        )
        return subset_db
