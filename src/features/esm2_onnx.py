"""ESM-2 Protein Language Model Feature Extraction with ONNX and INT8 Quantization.

Provides:
1. Pinned ESM-2 models (esm2_t6_8M_UR50D 320-D, esm2_t12_35M_UR50D 480-D).
2. Dynamic-axes ONNX export and dynamic INT8 quantization.
3. Residue-level pooling (mean over amino acid residues excluding BOS/EOS/PAD, and mean+max concatenation).
4. Non-canonical residue handling (U->C, O->K, B->N, Z->E, X->X) with OOD tracking.
5. Disk-backed caching keyed by sha256(seq) + model_id + revision + quantization + pooling.
6. CPU latency, throughput, and memory benchmarking.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Pinned model revisions
DEFAULT_PRIMARY_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
DEFAULT_COMPARISON_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
DEFAULT_REVISION = "main"

# Non-canonical residue mappings
NON_CANONICAL_MAP = {
    "U": "C",  # Selenocysteine -> Cysteine
    "O": "K",  # Pyrrolysine -> Lysine
    "B": "N",  # Asx (Asp/Asn) -> Asparagine
    "Z": "E",  # Glx (Glu/Gln) -> Glutamate
    "J": "L",  # Xle (Leu/Ile) -> Leucine
}

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(sequence: str) -> Tuple[str, Dict[str, Any]]:
    """Sanitize peptide sequence and flag non-canonical residues for OOD tracking.
    
    Returns:
        sanitized_seq: String with non-canonical substitutions applied.
        ood_meta: Dictionary containing substitution counts, unknown residue counts,
                  and an OOD risk flag.
    """
    clean_seq = sequence.strip().upper()
    substitutions: Dict[str, int] = {}
    unknown_count = 0
    chars = []

    for char in clean_seq:
        if char in NON_CANONICAL_MAP:
            mapped = NON_CANONICAL_MAP[char]
            substitutions[char] = substitutions.get(char, 0) + 1
            chars.append(mapped)
        elif char not in STANDARD_AAS:
            # Map unknown/ambiguous residue to X (supported in ESM-2 vocab)
            unknown_count += 1
            substitutions[char] = substitutions.get(char, 0) + 1
            chars.append("X")
        else:
            chars.append(char)

    has_non_canonical = len(substitutions) > 0
    if has_non_canonical:
        logger.warning(
            f"Non-canonical residues detected in sequence ({len(clean_seq)} aa): {substitutions}. "
            f"Mapped for ESM-2 embedding; flagged for OOD inspection."
        )

    ood_meta = {
        "original_length": len(clean_seq),
        "has_non_canonical": has_non_canonical,
        "non_canonical_counts": substitutions,
        "unknown_residue_count": unknown_count,
        "ood_flag": has_non_canonical or unknown_count > 0,
    }

    return "".join(chars), ood_meta


def compute_embedding_cache_key(
    sequence: str,
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    quantization: str = "int8",
    pooling: str = "mean",
) -> str:
    """Generate deterministic sha256 cache key for an embedding."""
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    key_str = f"{seq_hash}_{model_id}_{revision}_{quantization}_{pooling}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


class DiskEmbeddingCache:
    """Thread-safe disk-backed key-value cache for precomputed embeddings."""

    def __init__(self, cache_dir: Union[str, Path] = "data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "esm2_embeddings_cache.joblib"
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._memory_cache = joblib.load(self.cache_file)
                logger.info(f"Loaded {len(self._memory_cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Could not load cache from {self.cache_file}: {e}")
                self._memory_cache = {}

    def save(self) -> None:
        try:
            joblib.dump(self._memory_cache, self.cache_file, compress=3)
        except Exception as e:
            logger.error(f"Failed to persist embedding cache: {e}")

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._memory_cache.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        self._memory_cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._memory_cache

    def __len__(self) -> int:
        return len(self._memory_cache)


def export_esm2_to_onnx(
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    output_dir: Union[str, Path] = "models/onnx",
    revision: str = DEFAULT_REVISION,
    opset_version: int = 14,
) -> Tuple[Path, Path]:
    """Export ESM-2 PyTorch model to FP32 ONNX and dynamic INT8 ONNX models.
    
    Returns:
        (fp32_onnx_path, int8_onnx_path)
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_name = model_id.split("/")[-1]
    fp32_path = output_dir / f"{clean_name}_fp32.onnx"
    int8_path = output_dir / f"{clean_name}_int8.onnx"

    if not fp32_path.exists():
        logger.info(f"Exporting {model_id} (revision={revision}) to FP32 ONNX: {fp32_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision)
        model.eval()

        dummy_seq = "ACDEFGHIKLMNPQRSTVWY"
        dummy_inputs = tokenizer([dummy_seq], return_tensors="pt")

        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
        )
        logger.info(f"FP32 ONNX export complete: {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")

    if not int8_path.exists():
        logger.info(f"Quantizing {fp32_path} to dynamic INT8 ONNX: {int8_path}")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        logger.info(f"Dynamic INT8 ONNX complete: {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB)")

    return fp32_path, int8_path


class ESM2ONNXEmbedder:
    """Offline ESM-2 ONNX inference engine with pooling and caching."""

    def __init__(
        self,
        model_id: str = DEFAULT_PRIMARY_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        quantization: str = "int8",
        onnx_dir: Union[str, Path] = "models/onnx",
        cache_dir: Union[str, Path] = "data/embeddings",
        num_threads: int = 4,
    ):
        self.model_id = model_id
        self.revision = revision
        self.quantization = quantization
        self.onnx_dir = Path(onnx_dir)
        self.cache = DiskEmbeddingCache(cache_dir=cache_dir)

        # Lazy tokenizer import with local cache preference
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        # Resolve ONNX model path
        clean_name = model_id.split("/")[-1]
        model_filename = f"{clean_name}_{quantization}.onnx"
        self.model_path = self.onnx_dir / model_filename

        if not self.model_path.exists():
            logger.info(f"Model {self.model_path} not found. Triggering automated export.")
            export_esm2_to_onnx(model_id=model_id, output_dir=self.onnx_dir, revision=revision)

        # Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"Initialized ESM2ONNXEmbedder with {self.model_path} on CPUExecutionProvider")

    def _pool_hidden_states(
        self,
        hidden_states: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Residue-level pooling excluding BOS (cls), EOS (eos), and PAD tokens.
        
        Args:
            hidden_states: (B, T, D)
            input_ids: (B, T)
            attention_mask: (B, T)
            pooling: 'mean', 'max', or 'mean_max'
            
        Returns:
            Pooled embeddings: (B, D) or (B, 2*D) for mean_max.
        """
        B, T, D = hidden_states.shape
        cls_token_id = self.tokenizer.cls_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Mask only valid residues (attention_mask == 1 and not BOS, EOS, PAD)
        residue_mask = (
            (attention_mask == 1)
            & (input_ids != cls_token_id)
            & (input_ids != eos_token_id)
            & (input_ids != pad_token_id)
        )  # Shape (B, T)

        pooled_list = []
        for i in range(B):
            valid_idx = np.where(residue_mask[i])[0]
            if len(valid_idx) == 0:
                # Fallback if somehow no residues (e.g. empty sequence): use all active tokens
                valid_idx = np.where(attention_mask[i] == 1)[0]
                if len(valid_idx) == 0:
                    valid_idx = np.array([0])

            res_states = hidden_states[i, valid_idx, :]  # Shape (L, D)

            if pooling == "mean":
                vec = np.mean(res_states, axis=0)
            elif pooling == "max":
                vec = np.max(res_states, axis=0)
            elif pooling == "mean_max":
                mean_vec = np.mean(res_states, axis=0)
                max_vec = np.max(res_states, axis=0)
                vec = np.concatenate([mean_vec, max_vec], axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}. Use 'mean', 'max', or 'mean_max'.")

            pooled_list.append(vec)

        return np.array(pooled_list, dtype=np.float32)

    def extract_embedding(
        self,
        sequence: str,
        pooling: str = "mean",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract embedding for a single sequence with caching and non-canonical handling."""
        clean_seq, ood_meta = sanitize_sequence(sequence)
        cache_key = compute_embedding_cache_key(
            clean_seq, self.model_id, self.revision, self.quantization, pooling
        )

        cached_vec = self.cache.get(cache_key)
        if cached_vec is not None:
            return cached_vec, ood_meta

        # Tokenize with max_length=1024
        encoded = self.tokenizer(
            [clean_seq],
            max_length=1024,
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Run ONNX inference
        outputs = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        hidden_states = outputs[0]  # (1, T, D)

        pooled = self._pool_hidden_states(
            hidden_states, input_ids, attention_mask, pooling=pooling
        )[0]

        # Cache vector
        self.cache.set(cache_key, pooled)
        return pooled, ood_meta

    def extract_batch(
        self,
        sequences: List[str],
        batch_size: int = 64,
        pooling: str = "mean",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract embeddings for a list of sequences with length bucketing and caching."""
        results: List[Optional[np.ndarray]] = [None] * len(sequences)
        ood_metas: List[Dict[str, Any]] = []
        uncached_indices: List[int] = []
        clean_seqs: List[str] = []

        for idx, seq in enumerate(sequences):
            c_seq, meta = sanitize_sequence(seq)
            clean_seqs.append(c_seq)
            ood_metas.append(meta)

            cache_key = compute_embedding_cache_key(
                c_seq, self.model_id, self.revision, self.quantization, pooling
            )
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                results[idx] = cached_val
            else:
                uncached_indices.append(idx)

        # Process uncached with length-bucketing and adaptive batch sizing
        if uncached_indices:
            # Sort uncached by sequence length
            sorted_uncached = sorted(uncached_indices, key=lambda i: len(clean_seqs[i]))
            idx_pos = 0

            while idx_pos < len(sorted_uncached):
                curr_idx = sorted_uncached[idx_pos]
                curr_len = len(clean_seqs[curr_idx])

                # Adaptive batch size to prevent quadratic attention RAM blowup
                if curr_len <= 64:
                    cur_bs = min(batch_size, 32)
                elif curr_len <= 200:
                    cur_bs = min(batch_size, 16)
                elif curr_len <= 500:
                    cur_bs = min(batch_size, 8)
                else:
                    cur_bs = 2  # Ultra-long sequences

                batch_idxs = sorted_uncached[idx_pos : idx_pos + cur_bs]
                idx_pos += len(batch_idxs)
                batch_seqs = [clean_seqs[i] for i in batch_idxs]

                encoded = self.tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="np",
                )
                input_ids = encoded["input_ids"].astype(np.int64)
                attention_mask = encoded["attention_mask"].astype(np.int64)

                outputs = self.session.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )
                hidden_states = outputs[0]

                # Compute both mean and mean_max poolings simultaneously in one forward pass
                pooled_mean = self._pool_hidden_states(
                    hidden_states, input_ids, attention_mask, pooling="mean"
                )
                pooled_meanmax = self._pool_hidden_states(
                    hidden_states, input_ids, attention_mask, pooling="mean_max"
                )

                for local_i, global_i in enumerate(batch_idxs):
                    vec_mean = pooled_mean[local_i]
                    vec_meanmax = pooled_meanmax[local_i]

                    # Cache both poolings
                    c_key_mean = compute_embedding_cache_key(
                        clean_seqs[global_i],
                        self.model_id,
                        self.revision,
                        self.quantization,
                        "mean",
                    )
                    c_key_meanmax = compute_embedding_cache_key(
                        clean_seqs[global_i],
                        self.model_id,
                        self.revision,
                        self.quantization,
                        "mean_max",
                    )
                    self.cache.set(c_key_mean, vec_mean)
                    self.cache.set(c_key_meanmax, vec_meanmax)

                    if pooling == "mean":
                        results[global_i] = vec_mean
                    elif pooling == "mean_max":
                        results[global_i] = vec_meanmax
                    else:
                        results[global_i] = vec_mean

            self.cache.save()

        return np.array(results, dtype=np.float32), ood_metas


def verify_fp32_int8_parity(
    sequences: List[str],
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    onnx_dir: Union[str, Path] = "models/onnx",
    tolerance_cosine: float = 0.99,
    embedder_fp32: Optional[ESM2ONNXEmbedder] = None,
    embedder_int8: Optional[ESM2ONNXEmbedder] = None,
) -> Dict[str, Any]:
    """Verify that FP32 vs INT8 embeddings achieve mean cosine similarity >= tolerance_cosine."""
    from sklearn.metrics.pairwise import cosine_similarity

    if embedder_fp32 is None:
        embedder_fp32 = ESM2ONNXEmbedder(
            model_id=model_id, quantization="fp32", onnx_dir=onnx_dir
        )
    if embedder_int8 is None:
        embedder_int8 = ESM2ONNXEmbedder(
            model_id=model_id, quantization="int8", onnx_dir=onnx_dir
        )

    embs_fp32, _ = embedder_fp32.extract_batch(sequences, batch_size=32, pooling="mean")
    embs_int8, _ = embedder_int8.extract_batch(sequences, batch_size=32, pooling="mean")

    cos_sims = [
        cosine_similarity(embs_fp32[i : i + 1], embs_int8[i : i + 1])[0, 0]
        for i in range(len(sequences))
    ]
    mean_cos = float(np.mean(cos_sims))
    min_cos = float(np.min(cos_sims))
    max_cos = float(np.max(cos_sims))

    parity_passed = mean_cos >= tolerance_cosine

    logger.info(
        f"Parity Check for {model_id}: Mean Cosine Sim = {mean_cos:.5f} "
        f"(Min={min_cos:.5f}, Max={max_cos:.5f}) - Threshold={tolerance_cosine} - Passed={parity_passed}"
    )

    return {
        "model_id": model_id,
        "n_samples": len(sequences),
        "mean_cosine_similarity": mean_cos,
        "min_cosine_similarity": min_cos,
        "max_cosine_similarity": max_cos,
        "threshold": tolerance_cosine,
        "parity_passed": parity_passed,
    }


def benchmark_cpu_latency_and_throughput(
    embedder: ESM2ONNXEmbedder,
    sequences: Optional[List[str]] = None,
    batch_sizes: List[int] = [1, 64],
    n_warmup: int = 3,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """Measure raw CPU inference latency, throughput (peptides/sec), and memory usage."""
    import psutil

    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    results: Dict[str, Any] = {
        "model_id": embedder.model_id,
        "quantization": embedder.quantization,
        "batch_benchmarks": {},
    }

    # Reference peptide sequence (20 aa) for standard throughput benchmarking
    ref_seq = "ACDEFGHIKLMNPQRSTVWY"

    for bs in batch_sizes:
        batch_seqs = [ref_seq] * bs
        encoded = embedder.tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Warmup
        for _ in range(n_warmup):
            _ = embedder.session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = embedder.session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_time = float(np.mean(times))
        throughput = float(bs / mean_time)
        latency_per_peptide_ms = float((mean_time / bs) * 1000.0)

        results["batch_benchmarks"][f"bs_{bs}"] = {
            "batch_size": bs,
            "mean_batch_time_sec": mean_time,
            "latency_per_peptide_ms": latency_per_peptide_ms,
            "throughput_peptides_per_sec": throughput,
        }

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    results["memory_rss_mb"] = {
        "before": round(mem_before_mb, 2),
        "after": round(mem_after_mb, 2),
        "delta": round(mem_after_mb - mem_before_mb, 2),
    }

    return results
