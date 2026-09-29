"""Aggregator module for Peptide Profiler.

Merges physicochemical, antigenicity, allergenicity, and toxicity predictions
into a unified nature profile per sequence and exports CSV, JSON, and HTML reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.allergenicity import run_allergenicity
from src.antigenicity import run_antigenicity
from src.physicochem import run_physicochemical
from src.toxicity import run_toxicity


def predict_bcell_epitopes(sequence: str, window_size: int = 7, threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Predict linear B-cell epitopes using the Kolaskar-Tongaonkar antigenicity scale.

    Args:
        sequence: Amino acid sequence.
        window_size: Sliding window size (default 7).
        threshold: Antigenic threshold (default 1.0, values > 1.0 are antigenic).

    Returns:
        List of identified epitope regions with start, end, sub-sequence, and average score.
    """
    # Kolaskar & Tongaonkar antigenicity values for 20 amino acids
    kt_scale = {
        "A": 1.064, "C": 1.412, "D": 0.866, "E": 0.851, "F": 1.091,
        "G": 0.874, "H": 1.105, "I": 1.152, "K": 0.930, "L": 1.250,
        "M": 0.826, "N": 0.776, "P": 1.064, "Q": 1.015, "R": 0.952,
        "S": 1.012, "T": 0.909, "V": 1.383, "W": 0.893, "Y": 0.923,
    }
    n = len(sequence)
    if n < window_size:
        return []

    # Calculate window averages
    scores = []
    for i in range(n - window_size + 1):
        window = sequence[i : i + window_size]
        avg = sum(kt_scale.get(aa, 1.0) for aa in window) / window_size
        scores.append((i, i + window_size, avg))

    # Merge overlapping windows exceeding threshold
    epitopes = []
    current_start = None
    current_end = None
    window_scores = []

    for start, end, sc in scores:
        if sc >= threshold:
            if current_start is None:
                current_start = start
                current_end = end
                window_scores = [sc]
            else:
                current_end = end
                window_scores.append(sc)
        else:
            if current_start is not None:
                epitopes.append({
                    "start": current_start + 1,  # 1-indexed
                    "end": current_end,
                    "length": current_end - current_start,
                    "sequence": sequence[current_start:current_end],
                    "mean_score": round(sum(window_scores) / len(window_scores), 3),
                })
                current_start = None
                current_end = None
                window_scores = []

    if current_start is not None:
        epitopes.append({
            "start": current_start + 1,
            "end": current_end,
            "length": current_end - current_start,
            "sequence": sequence[current_start:current_end],
            "mean_score": round(sum(window_scores) / len(window_scores), 3),
        })

    return epitopes


# Config path for business cost matrices and calibrated thresholds
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "screening_profiles.json"
_CACHED_SCREENING_CONFIG = None


def load_screening_config() -> Optional[Dict[str, Any]]:
    """Load versioned screening profiles and cost-optimal thresholds."""
    global _CACHED_SCREENING_CONFIG
    if _CACHED_SCREENING_CONFIG is not None:
        return _CACHED_SCREENING_CONFIG

    if CONFIG_PATH.exists():
        try:
            _CACHED_SCREENING_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return _CACHED_SCREENING_CONFIG
        except Exception:
            pass
    return None


def get_profile_thresholds(candidate_type: str = "vaccine", length: int = 100) -> Dict[str, Any]:
    """Retrieve cost-optimal thresholds and expected cost per length bin from config."""
    cfg = load_screening_config()
    ctype = candidate_type.lower()

    defaults = {
        "tox_threshold": 0.60,
        "alg_threshold": 0.50,
        "ant_threshold": 0.50,
        "expected_cost_usd": 500.0,
    }
    if not cfg or "profiles" not in cfg or ctype not in cfg["profiles"]:
        return defaults

    prof_data = cfg["profiles"][ctype].get("optimal_thresholds", {})

    if length < 15:
        bin_key = "< 15 aa"
    elif length <= 50:
        bin_key = "15-50 aa"
    elif length <= 200:
        bin_key = "50-200 aa"
    elif length <= 500:
        bin_key = "200-500 aa"
    else:
        bin_key = "> 500 aa"

    tox_th = prof_data.get("toxicity", {}).get(bin_key, {}).get("threshold", 0.60)
    alg_th = prof_data.get("allergenicity", {}).get(bin_key, {}).get("threshold", 0.50)
    ant_th = prof_data.get("antigenicity", {}).get(bin_key, {}).get("threshold", 0.50)
    exp_cost = prof_data.get("toxicity", {}).get(bin_key, {}).get("expected_cost_per_lead_usd", 500.0)

    return {
        "tox_threshold": tox_th,
        "alg_threshold": alg_th,
        "ant_threshold": ant_th,
        "expected_cost_usd": exp_cost,
    }


import math


def calibrate_probability(raw_score: float, task: str = "toxicity") -> float:
    """Transform raw model output score to calibrated probability via frozen Platt scalers."""
    cfg = load_screening_config()
    if not cfg or "metadata" not in cfg or "platt_scalers" not in cfg["metadata"]:
        return raw_score
    scaler = cfg["metadata"]["platt_scalers"].get(task)
    if not scaler:
        return raw_score

    coef = scaler.get("coef", 1.0)
    intercept = scaler.get("intercept", 0.0)

    eps = 1e-6
    clipped = max(eps, min(1.0 - eps, raw_score))
    log_odds = math.log(clipped / (1.0 - clipped))
    z = (coef * log_odds) + intercept
    cal_p = 1.0 / (1.0 + math.exp(-z))
    return float(round(max(0.0, min(1.0, cal_p)), 4))


def get_provenance_stamp() -> Dict[str, Any]:
    """Retrieve cryptographic model hashes, calibration set ID, and feature extractor versions."""
    scr_cfg = load_screening_config()
    cal_id = "swiss_prot_eval_v1_437_clusters"
    if scr_cfg and "metadata" in scr_cfg:
        cal_id = scr_cfg["metadata"].get("calibration_set_id", cal_id)

    return {
        "pipeline_version": "2.1.0",
        "calibration_set_id": cal_id,
        "feature_extractor_version": "pfeature_pcp_aac_v1.0",
        "cluster_artifact_version": "cluster_assignments_v1_35pct",
        "model_sha256_hashes": {
            "toxicity": "5683cd21b20c627c09dc85c2d3ad6d33533003fc01f4acae5b444368bb3e2bde",
            "antigenicity": "c1ee5bf6aac96b3236d2e91493de12a7dfcdd834985e9ff18cdc4ff780890d60",
            "allergenicity": "b1cbddfe919b01483cc37483c115243efc3ddefe84a6f9e565eab86f14d498fc",
        },
    }


def calculate_desirability(
    antigenicity_score: float,
    toxicity_score: float,
    allergenicity_score: float,
    candidate_type: str = "vaccine",
) -> float:
    """Calculate multi-objective candidate desirability score in [0.0, 1.0].

    For vaccines: High antigenicity, low toxicity, low allergenicity.
    For therapeutics: Low antigenicity (non-immunogenic), low toxicity, low allergenicity.
    """
    if candidate_type.lower() == "therapeutic":
        # Desirability = (1 - antigenicity) * (1 - toxicity) * (1 - allergenicity)
        score = (1.0 - antigenicity_score) * (1.0 - toxicity_score) * (1.0 - allergenicity_score)
    else:
        # Vaccine: Desirability = antigenicity * (1 - toxicity) * (1 - allergenicity)
        score = antigenicity_score * (1.0 - toxicity_score) * (1.0 - allergenicity_score)

    return round(float(max(0.0, min(1.0, score))), 4)


def profile_sequence(
    record: Dict[str, Any],
    organism_type: str = "bacteria",
    tox_threshold: Optional[float] = None,
    alg_threshold: Optional[float] = None,
    ant_threshold: Optional[float] = None,
    include_epitopes: bool = True,
    use_docker: bool = False,
    use_api: bool = False,
    candidate_type: str = "vaccine",
    enforce_safety_gates: bool = True,
) -> Dict[str, Any]:
    """Execute complete characterization pipeline with hard safety gates and uncertainty bounds."""
    seq_id = str(record["id"])
    seq = str(record["sequence"])
    seq_len = len(seq)

    # Resolve config-driven thresholds if not explicitly passed
    cfg_thresh = get_profile_thresholds(candidate_type=candidate_type, length=seq_len)
    active_tox_th = tox_threshold if tox_threshold is not None else cfg_thresh["tox_threshold"]
    active_alg_th = alg_threshold if alg_threshold is not None else cfg_thresh["alg_threshold"]
    active_ant_th = ant_threshold if ant_threshold is not None else cfg_thresh["ant_threshold"]
    exp_cost = cfg_thresh.get("expected_cost_usd", 0.0)

    physico = run_physicochemical(seq)
    antigen = run_antigenicity(
        seq,
        organism_type=organism_type,
        threshold=active_ant_th,
        use_docker=use_docker,
    )
    toxic = run_toxicity(seq, threshold=active_tox_th)
    allergen = run_allergenicity(seq, threshold=active_alg_th, use_api=use_api)

    # Compute calibrated probabilities via frozen Platt scalers
    raw_tox = toxic["toxicity_score"]
    raw_alg = allergen["allergenicity_score"]
    raw_ant = antigen["antigenicity_score"]

    cal_tox = calibrate_probability(raw_tox, "toxicity")
    cal_alg = calibrate_probability(raw_alg, "allergenicity")
    cal_ant = calibrate_probability(raw_ant, "antigenicity")

    # Hard Safety Gate Evaluation & Reason Codes
    reason_codes: List[str] = []
    abstentions: List[str] = []

    if seq_len < 15:
        abstentions.append("LOW_CONFIDENCE_SHORT_PEPTIDE")

    is_toxic = bool((cal_tox >= active_tox_th) or (raw_tox >= 0.60))
    is_allergen = bool((cal_alg >= active_alg_th) or (raw_alg >= 0.65))
    is_antigen = bool((cal_ant >= active_ant_th) or (raw_ant >= 0.50))

    if is_toxic:
        reason_codes.append("SAFETY_VIOLATION_TOXICITY")
    if is_allergen:
        reason_codes.append("SAFETY_VIOLATION_ALLERGENICITY")

    if candidate_type.lower() == "therapeutic" and is_antigen:
        reason_codes.append("IMMUNOGENICITY_RISK")

    # Hard Safety Gate vs Flagging Logic
    # 1. Any candidate with confirmed toxicity is strictly EXCLUDED
    # 2. In therapeutic profile, combined allergenicity + immunogenicity is EXCLUDED
    # 3. Severe allergenicity (raw >= 0.70) is EXCLUDED
    # 4. Moderate/single warnings are FLAGGED (survivors ranked with conservative penalty)
    is_excluded = False
    if enforce_safety_gates:
        if "SAFETY_VIOLATION_TOXICITY" in reason_codes:
            is_excluded = True
        elif candidate_type.lower() == "therapeutic" and (
            "SAFETY_VIOLATION_ALLERGENICITY" in reason_codes and "IMMUNOGENICITY_RISK" in reason_codes
        ):
            is_excluded = True
        elif "SAFETY_VIOLATION_ALLERGENICITY" in reason_codes and raw_alg >= 0.70:
            is_excluded = True

    if is_excluded:
        candidate_status = "EXCLUDED"
        desirability_score = 0.0
        conservative_rank_score = 0.0
    else:
        candidate_status = "FLAGGED" if reason_codes else "APPROVED"
        # Compute multi-objective desirability from CALIBRATED probabilities
        desirability_score = calculate_desirability(
            antigenicity_score=cal_ant,
            toxicity_score=cal_tox,
            allergenicity_score=cal_alg,
            candidate_type=candidate_type,
        )
        # Conservative ranking bound penalizing residual risk and uncertainty
        penalty = 0.10 * (cal_tox + cal_alg)
        conservative_rank_score = round(float(max(0.0, desirability_score - penalty)), 4)

    epitopes = predict_bcell_epitopes(seq) if include_epitopes else []

    profile = {
        "id": seq_id,
        "sequence": seq,
        "candidate_status": candidate_status,
        "reason_codes": reason_codes,
        "abstentions": abstentions,
        "provenance": get_provenance_stamp(),
        "antigenicity": {
            "score": antigen["antigenicity_score"],
            "calibrated_probability": cal_ant,
            "is_antigen": is_antigen,
            "method": antigen.get("method", "VaxiJen-ML"),
            "threshold": active_ant_th,
        },
        "allergenicity": {
            "score": allergen["allergenicity_score"],
            "calibrated_probability": cal_alg,
            "is_allergen": is_allergen,
            "method": allergen.get("method", "AlgPred/AllerTOP"),
            "threshold": active_alg_th,
        },
        "toxicity": {
            "score": toxic["toxicity_score"],
            "calibrated_probability": cal_tox,
            "is_toxic": is_toxic,
            "method": toxic.get("method", "ToxinPred2"),
            "threshold": active_tox_th,
        },
        "physicochemical": {
            "mol_weight": physico["mol_weight"],
            "gravy": physico["gravy"],
            "instability_index": physico["instability_index"],
            "is_stable": physico["is_stable"],
            "isoelectric_point": physico["isoelectric_point"],
            "charge_at_pH7": physico["charge_at_pH7"],
            "aromaticity": physico["aromaticity"],
            "secondary_structure": physico["secondary_structure"],
        },
        "desirability": {
            "score": desirability_score,
            "conservative_rank_score": conservative_rank_score,
            "candidate_type": candidate_type,
            "expected_cost_usd": exp_cost,
        },
        "epitopes": {
            "count": len(epitopes),
            "regions": epitopes,
        },
    }

    return profile


def profile_multiple_sequences(
    records: List[Dict[str, Any]],
    organism_type: str = "bacteria",
    tox_threshold: Optional[float] = None,
    alg_threshold: Optional[float] = None,
    ant_threshold: Optional[float] = None,
    candidate_type: str = "vaccine",
    rank_candidates: bool = True,
    use_docker: bool = False,
    use_api: bool = False,
    enforce_safety_gates: bool = True,
) -> List[Dict[str, Any]]:
    """Profile a batch of peptide sequences with hard safety gating and conservative ranking."""
    profiles: List[Dict[str, Any]] = []

    for rec in records:
        prof = profile_sequence(
            rec,
            organism_type=organism_type,
            tox_threshold=tox_threshold,
            alg_threshold=alg_threshold,
            ant_threshold=ant_threshold,
            use_docker=use_docker,
            use_api=use_api,
            candidate_type=candidate_type,
            enforce_safety_gates=enforce_safety_gates,
        )
        profiles.append(prof)

    if rank_candidates:
        # Separate candidate survivors (APPROVED and FLAGGED) from EXCLUDED safety violations
        survivors = [p for p in profiles if p["candidate_status"] != "EXCLUDED"]
        excluded = [p for p in profiles if p["candidate_status"] == "EXCLUDED"]

        # Sort survivors by conservative rank score descending
        survivors.sort(key=lambda x: x["desirability"]["conservative_rank_score"], reverse=True)

        # Sort excluded candidates by toxicity risk ascending
        excluded.sort(key=lambda x: x["toxicity"]["score"])

        ordered_profiles = survivors + excluded
        for rank, p in enumerate(ordered_profiles, start=1):
            p["desirability"]["rank"] = rank

        return ordered_profiles

    return profiles


def to_dataframe(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of sequence profiles into a flattened pandas DataFrame."""
    rows = []
    for p in profiles:
        row = {
            "ID": p["id"],
            "Length": len(p["sequence"]),
            "Sequence": p["sequence"],
            "Antigenicity_Score": p["antigenicity"]["score"],
            "Calibrated_Antigenicity": p["antigenicity"].get("calibrated_probability", p["antigenicity"]["score"]),
            "Is_Antigen": p["antigenicity"]["is_antigen"],
            "Allergenicity_Score": p["allergenicity"]["score"],
            "Calibrated_Allergenicity": p["allergenicity"].get("calibrated_probability", p["allergenicity"]["score"]),
            "Is_Allergen": p["allergenicity"]["is_allergen"],
            "Toxicity_Score": p["toxicity"]["score"],
            "Calibrated_Toxicity": p["toxicity"].get("calibrated_probability", p["toxicity"]["score"]),
            "Is_Toxic": p["toxicity"]["is_toxic"],
            "Candidate_Status": p.get("candidate_status", "APPROVED"),
            "Reason_Codes": ", ".join(p.get("reason_codes", [])) if p.get("reason_codes") else "PASS",
            "Abstentions": ", ".join(p.get("abstentions", [])) if p.get("abstentions") else "NONE",
            "Desirability_Score": p["desirability"]["score"],
            "Conservative_Rank_Score": p["desirability"].get("conservative_rank_score", p["desirability"]["score"]),
            "Desirability_Rank": p["desirability"].get("rank", "-"),
            "Expected_Cost_USD": p["desirability"].get("expected_cost_usd", 0.0),
            "Mol_Weight_Da": p["physicochemical"]["mol_weight"],
            "GRAVY": p["physicochemical"]["gravy"],
            "Instability_Index": p["physicochemical"]["instability_index"],
            "Is_Stable": p["physicochemical"]["is_stable"],
            "Isoelectric_Point": p["physicochemical"]["isoelectric_point"],
            "Net_Charge_pH7": p["physicochemical"]["charge_at_pH7"],
            "Aromaticity": p["physicochemical"]["aromaticity"],
            "Helix_Fraction": p["physicochemical"]["secondary_structure"]["helix"],
            "Sheet_Fraction": p["physicochemical"]["secondary_structure"]["sheet"],
            "Turn_Fraction": p["physicochemical"]["secondary_structure"]["turn"],
            "Epitopes_Count": p["epitopes"]["count"],
            "Antigen_Method": p["antigenicity"]["method"],
            "Toxin_Method": p["toxicity"]["method"],
            "Allergen_Method": p["allergenicity"]["method"],
            "Model_Tox_Hash": p.get("provenance", {}).get("model_sha256_hashes", {}).get("toxicity", "5683cd21b20c")[:12],
            "Model_Alg_Hash": p.get("provenance", {}).get("model_sha256_hashes", {}).get("allergenicity", "b1cbddfe919b")[:12],
            "Model_Ant_Hash": p.get("provenance", {}).get("model_sha256_hashes", {}).get("antigenicity", "c1ee5bf6aac9")[:12],
            "Calibration_Set_ID": p.get("provenance", {}).get("calibration_set_id", "swiss_prot_eval_v1_437_clusters"),
            "Feature_Extractor_Version": p.get("provenance", {}).get("feature_extractor_version", "pfeature_pcp_aac_v1.0"),
            "Cluster_Artifact_Version": p.get("provenance", {}).get("cluster_artifact_version", "cluster_assignments_v1_35pct"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def export_reports(
    profiles: List[Dict[str, Any]],
    output_prefix: Union[str, Path],
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Export profiles to CSV, JSON, and interactive HTML.

    Args:
        profiles: List of sequence profile dictionaries.
        output_prefix: Base path without extension or with extension.
        formats: List of formats to write, e.g. ['csv', 'json', 'html']. Defaults to all.

    Returns:
        Dict mapping format name to generated file Path.
    """
    if formats is None:
        formats = ["csv", "json", "html"]

    prefix = Path(output_prefix)
    if prefix.suffix in [".csv", ".json", ".html"]:
        base = prefix.with_suffix("")
    else:
        base = prefix

    base.parent.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}
    df = to_dataframe(profiles)

    # 1. JSON
    if "json" in formats:
        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        generated["json"] = json_path

    # 2. CSV
    if "csv" in formats:
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        generated["csv"] = csv_path

    # 3. HTML Interactive Report
    if "html" in formats:
        html_path = base.with_suffix(".html")
        html_content = generate_html_report(profiles, df)
        html_path.write_text(html_content, encoding="utf-8")
        generated["html"] = html_path

    return generated


def generate_html_report(profiles: List[Dict[str, Any]], df: pd.DataFrame) -> str:
    """Generate a sleek, modern, self-contained HTML characterization report."""
    total_seqs = len(profiles)
    antigens = sum(1 for p in profiles if p["antigenicity"]["is_antigen"])
    toxins = sum(1 for p in profiles if p["toxicity"]["is_toxic"])
    allergens = sum(1 for p in profiles if p["allergenicity"]["is_allergen"])

    # Table rows
    table_rows = []
    for p in profiles:
        ant_badge = (
            f'<span class="badge badge-success">Antigen ({p["antigenicity"]["score"]:.2f})</span>'
            if p["antigenicity"]["is_antigen"]
            else f'<span class="badge badge-muted">Non-antigen ({p["antigenicity"]["score"]:.2f})</span>'
        )
        tox_badge = (
            f'<span class="badge badge-danger">Toxic ({p["toxicity"]["score"]:.2f})</span>'
            if p["toxicity"]["is_toxic"]
            else f'<span class="badge badge-success">Non-toxic ({p["toxicity"]["score"]:.2f})</span>'
        )
        alg_badge = (
            f'<span class="badge badge-warning">Allergen ({p["allergenicity"]["score"]:.2f})</span>'
            if p["allergenicity"]["is_allergen"]
            else f'<span class="badge badge-success">Non-allergen ({p["allergenicity"]["score"]:.2f})</span>'
        )
        rank_val = p["desirability"].get("rank", "-")
        des_score = p["desirability"]["score"]
        ep_count = p["epitopes"]["count"]

        table_rows.append(f"""
        <tr>
            <td><strong>#{rank_val}</strong></td>
            <td><code>{p["id"]}</code></td>
            <td class="seq-cell" title="{p["sequence"]}">{p["sequence"]}</td>
            <td>{p["physicochemical"]["mol_weight"]} Da</td>
            <td>{p["physicochemical"]["gravy"]}</td>
            <td>{p["physicochemical"]["instability_index"]} ({'Stable' if p["physicochemical"]["is_stable"] else 'Unstable'})</td>
            <td>{ant_badge}</td>
            <td>{tox_badge}</td>
            <td>{alg_badge}</td>
            <td><strong>{des_score:.4f}</strong></td>
            <td><span class="badge badge-info">{ep_count} Epitopes</span></td>
        </tr>
        """)

    table_body = "\n".join(table_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Peptide Profiler Report (VaxiJen-Alternative)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --surface: #161b22;
            --surface-hover: #1c2128;
            --border: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #238636;
            --warning: #d29922;
            --danger: #da3633;
            --info: #1f6feb;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, sans-serif;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            margin-top: 4px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent);
        }}
        .metric-title {{
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            margin-top: 8px;
            color: #ffffff;
        }}
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow-x: auto;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}
        th {{
            background: rgba(22, 27, 34, 0.95);
            padding: 14px 16px;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{
            background: var(--surface-hover);
        }}
        .seq-cell {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #79c0ff;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        }}
        .badge-success {{ background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }}
        .badge-danger {{ background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #da3633; }}
        .badge-warning {{ background: rgba(210, 153, 34, 0.2); color: #e3b341; border: 1px solid #d29922; }}
        .badge-info {{ background: rgba(31, 111, 235, 0.2); color: #58a6ff; border: 1px solid #1f6feb; }}
        .badge-muted {{ background: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid #30363d; }}
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: rgba(110, 118, 129, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }}
        footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Peptide Characterization Report</h1>
                <div class="subtitle">Open-Source Pipeline: Antigenicity (VaxiJen-alt) • Toxicity (ToxinPred2) • Allergenicity (AllerTOP-alt) • Physicochemical Properties</div>
            </div>
            <div>
                <span class="badge badge-info">Candidates: {total_seqs}</span>
            </div>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Total Sequences</div>
                <div class="metric-value">{total_seqs}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Antigenic Peptides</div>
                <div class="metric-value" style="color: #3fb950;">{antigens}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Non-Toxic Peptides</div>
                <div class="metric-value" style="color: #58a6ff;">{total_seqs - toxins}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Non-Allergenic Peptides</div>
                <div class="metric-value" style="color: #e3b341;">{total_seqs - allergens}</div>
            </div>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>ID</th>
                        <th>Sequence</th>
                        <th>Mol Wt</th>
                        <th>GRAVY</th>
                        <th>Stability</th>
                        <th>Antigenicity</th>
                        <th>Toxicity</th>
                        <th>Allergenicity</th>
                        <th>Desirability</th>
                        <th>Epitopes</th>
                    </tr>
                </thead>
                <tbody>
                    {table_body}
                </tbody>
            </table>
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 30px; font-size: 13px;">
            <h3 style="color: var(--accent); margin-bottom: 10px; font-size: 15px;">Provenance & Audit Stamp</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; color: var(--text-secondary);">
                <div><strong>Calibration Set ID:</strong> <code>swiss_prot_eval_v1_437_clusters</code></div>
                <div><strong>Cluster Version:</strong> <code>cluster_assignments_v1_35pct</code></div>
                <div><strong>Feature Extractor:</strong> <code>pfeature_pcp_aac_v1.0</code></div>
                <div><strong>Toxicity Model (ONNX):</strong> <code>5683cd21b20c627c...</code></div>
                <div><strong>Allergen Model (RF):</strong> <code>b1cbddfe919b0148...</code></div>
                <div><strong>Antigen Model (RF):</strong> <code>c1ee5bf6aac96b32...</code></div>
            </div>
        </div>

        <footer>
            Generated by Open-Source Peptide Characterization Pipeline • Powered by Biopython, Pfeature, ToxinPred2 & VaxiJen ML
        </footer>
    </div>
</body>
</html>
"""
