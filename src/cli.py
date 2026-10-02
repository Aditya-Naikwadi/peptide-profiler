"""Command-Line Interface (CLI) for Peptide Profiler."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path regardless of execution method
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aggregator import export_reports, profile_multiple_sequences, to_dataframe
from src.parser import SequenceValidationError, parse_fasta


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="peptide-profiler",
        description=(
            "Open-Source Peptide Characterization Pipeline (VaxiJen-alternative).\n"
            "Calculates Antigenicity, Allergenicity, Toxicity, and Physicochemical profiles "
            "from FASTA input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input FASTA file path containing one or more peptide sequences, or raw sequence.",
    )
    parser.add_argument(
        "-o", "--output",
        default="report.csv",
        help="Output file path (e.g. report.csv, report.json, report.html, or directory). Default: report.csv",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["csv", "json", "html", "all"],
        default="all",
        help="Report export format: csv, json, html, or all (default: all).",
    )
    parser.add_argument(
        "--organism",
        default="bacteria",
        choices=["bacteria", "virus", "parasite", "fungus", "general"],
        help="Organism context for antigenicity prediction (default: bacteria).",
    )
    parser.add_argument(
        "--tox-thresh",
        type=float,
        default=0.6,
        help="Toxicity classification threshold (default: 0.6 per ToxinPred2).",
    )
    parser.add_argument(
        "--ant-thresh",
        type=float,
        default=0.5,
        help="Antigenicity classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--alg-thresh",
        type=float,
        default=0.5,
        help="Allergenicity classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--candidate-type",
        choices=["vaccine", "therapeutic"],
        default="vaccine",
        help="Candidate design goal: 'vaccine' (maximize antigenicity) or 'therapeutic' (minimize antigenicity).",
    )
    parser.add_argument(
        "--profile",
        choices=["vaccine", "therapeutic"],
        default=None,
        help="Screening profile: 'vaccine' or 'therapeutic' (alias/override for --candidate-type).",
    )
    parser.add_argument(
        "--target-prevalence",
        type=float,
        default=0.02,
        help="Target screening prevalence (pi_t) for Bayes prior adjustment (default: 0.02).",
    )
    parser.add_argument(
        "--standard-contract",
        action="store_true",
        default=False,
        help="Export JSON conforming strictly to the unified output schema contract.",
    )
    parser.add_argument(
        "--no-rank",
        action="store_true",
        help="Disable ranking candidates by combined desirability score.",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Automatically sanitize non-standard amino acid characters instead of erroring.",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Attempt to run Vaxign-ML via Docker container if Docker is installed and running.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Attempt to query remote AllerTOP / AlgPred2 web APIs if internet is reachable.",
    )
    return parser


def print_banner():
    banner = """
========================================================================
   PEPTIDE PROFILER • Open-Source Peptide Characterization Pipeline
   VaxiJen-Alternative • ToxinPred2 • AllerTOP-Alt • Physicochemical
========================================================================
    """
    print(banner)


def main(argv: List[str] = None):
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    print_banner()

    start_time = time.time()
    input_path = Path(args.input)

    # 1. Parse input
    print(f"[*] Reading input: {args.input}")
    try:
        if input_path.exists():
            records = parse_fasta(input_path, sanitize=args.sanitize)
        else:
            # Check if user passed raw sequence or FASTA directly
            records = parse_fasta(args.input, is_content=True, sanitize=args.sanitize)
    except SequenceValidationError as e:
        print(f"[!] Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[!] Failed to parse input: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Loaded {len(records)} sequence(s) successfully.")

    active_profile = args.profile if args.profile is not None else args.candidate_type
    target_prev = float(args.target_prevalence)

    # 2. Run profiling
    print(f"[*] Characterizing peptides (Profile: {active_profile.upper()}, Target Prev: {target_prev:.2%})...")
    profiles = profile_multiple_sequences(
        records=records,
        organism_type=args.organism,
        tox_threshold=args.tox_thresh,
        alg_threshold=args.alg_thresh,
        ant_threshold=args.ant_thresh,
        candidate_type=active_profile,
        rank_candidates=not args.no_rank,
        use_docker=args.use_docker,
        use_api=args.use_api,
    )

    # 3. Determine formats
    if args.format == "all":
        formats = ["csv", "json", "html"]
    else:
        formats = [args.format]

    # If user provided a specific extension in -o (e.g. -o output.csv), ensure that format is included
    out_path = Path(args.output)
    if out_path.suffix.lower() == ".csv" and "csv" not in formats:
        formats.append("csv")
    elif out_path.suffix.lower() == ".json" and "json" not in formats:
        formats.append("json")
    elif out_path.suffix.lower() == ".html" and "html" not in formats:
        formats.append("html")

    # 4. Export reports
    generated = export_reports(profiles, output_prefix=args.output, formats=formats)

    # Export standard contract JSON
    if "json" in formats:
        from src.api import serialize_to_standard_contract
        import json as json_lib

        standard_contracts = [
            serialize_to_standard_contract(
                p,
                profile_type=active_profile,
                target_prevalence=target_prev,
            )
            for p in profiles
        ]
        contract_data = standard_contracts[0] if len(standard_contracts) == 1 else standard_contracts

        base_path = out_path.with_suffix("") if out_path.suffix.lower() in [".csv", ".json", ".html"] else out_path
        contract_json_path = base_path.with_name(f"{base_path.name}.contract.json")
        contract_json_path.write_text(json_lib.dumps(contract_data, indent=2), encoding="utf-8")
        generated["contract_json"] = contract_json_path

        if args.standard_contract:
            # Overwrite main json with the exact standard contract
            generated["json"].write_text(json_lib.dumps(contract_data, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"\n[+] Analysis complete in {elapsed:.2f} seconds.")
    print("[*] Generated reports:")
    for fmt, fpath in generated.items():
        print(f"    - {fmt.upper()}: {fpath.resolve()}")

    # 5. Print summary table to stdout
    df = to_dataframe(profiles)
    summary_cols = [
        "Desirability_Rank", "ID", "Length", "Mol_Weight_Da",
        "Antigenicity_Score", "Is_Antigen",
        "Toxicity_Score", "Is_Toxic",
        "Allergenicity_Score", "Is_Allergen",
        "Desirability_Score", "Is_Stable",
    ]
    avail_cols = [c for c in summary_cols if c in df.columns]

    print("\n" + "=" * 90)
    print(" CANDIDATE SUMMARY TABLE")
    print("=" * 90)
    print(df[avail_cols].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
