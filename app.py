"""Streamlit interactive web dashboard for Peptide Profiler."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.aggregator import (
    calculate_desirability,
    export_reports,
    predict_bcell_epitopes,
    profile_multiple_sequences,
    to_dataframe,
)
from src.parser import SequenceValidationError, parse_fasta

# Page configuration
st.set_page_config(
    page_title="Peptide Profiler • Nature Profile",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark theme
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #333f50;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-lbl {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb, #238636);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #388bfd, #2ea043);
    }
</style>
""", unsafe_allow_html=True)


def load_sample_fasta() -> str:
    sample_file = Path(__file__).parent / "examples" / "sample_peptides.fa"
    if sample_file.exists():
        return sample_file.read_text(encoding="utf-8")
    return ">OspA_Antigen\nMKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK\n>Melittin_Toxin\nGIGAVLKVLTTGLPALISWIKRKRQQ\n"


def main():
    st.title("🧬 Open-Source Peptide Characterization Pipeline")
    st.caption("Consolidated Nature Profile: Antigenicity (VaxiJen-alt) • Toxicity (ToxinPred2) • Allergenicity (AllerTOP-alt) • Physicochemical Properties")

    # Automatically pre-populate with benchmark results on first load so dashboard isn't blank
    if "profiles" not in st.session_state:
        cached_report = Path(__file__).parent / "examples" / "report.json"
        if cached_report.exists():
            try:
                st.session_state["profiles"] = json.loads(cached_report.read_text(encoding="utf-8"))
            except Exception:
                st.session_state["profiles"] = None

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        input_source = st.radio(
            "Input Mode",
            ["Sample Benchmark Peptides", "Upload FASTA", "Paste Sequence / FASTA"],
        )

        candidate_type = st.selectbox(
            "Candidate Goal",
            ["vaccine", "therapeutic"],
            help="Vaccine candidates maximize antigenicity; therapeutics minimize it.",
        )

        organism_type = st.selectbox(
            "Organism Context",
            ["bacteria", "virus", "parasite", "fungus", "general"],
        )

        st.subheader("Decision Thresholds")
        tox_thresh = st.slider("Toxicity Cutoff", 0.0, 1.0, 0.60, 0.05)
        ant_thresh = st.slider("Antigenicity Cutoff", 0.0, 1.0, 0.50, 0.05)
        alg_thresh = st.slider("Allergenicity Cutoff", 0.0, 1.0, 0.50, 0.05)

        st.subheader("Advanced Engine")
        use_docker = st.checkbox("Use Vaxign-ML Docker (if available)", value=False)
        use_api = st.checkbox("Query Web APIs (AlgPred2/AllerTOP)", value=False)
        sanitize = st.checkbox("Sanitize non-standard residues", value=True)

    # Input handling
    fasta_text = ""
    if input_source == "Sample Benchmark Peptides":
        fasta_text = load_sample_fasta()
        st.info("Loaded 8 curated benchmark peptides (protective antigens, allergens, toxins, non-toxic controls).")
        with st.expander("View FASTA Sequences"):
            st.code(fasta_text, language="fasta")

    elif input_source == "Upload FASTA":
        uploaded = st.file_uploader("Upload FASTA file (.fa, .fasta, .txt)", type=["fa", "fasta", "txt"])
        if uploaded:
            fasta_text = uploaded.getvalue().decode("utf-8")
            st.success(f"Loaded uploaded file ({len(fasta_text)} bytes)")
    else:
        fasta_text = st.text_area(
            "Paste Sequence or FASTA content here",
            value=">Peptide_Candidate_1\nALWKTLLKKVLKAAAKA\n>Peptide_Candidate_2\nGIGAVLKVLTTGLPALISWIKRKRQQ",
            height=150,
        )

    # Run profiling
    if st.button("🚀 Run Full Characterization Pipeline", use_container_width=True):
        if not fasta_text.strip():
            st.error("Please provide valid FASTA sequences.")
            return

        with st.spinner("Parsing and validating sequences..."):
            try:
                records = parse_fasta(fasta_text, is_content=True, sanitize=sanitize)
            except SequenceValidationError as e:
                st.error(f"Sequence Validation Error: {e}")
                return
            except Exception as e:
                st.error(f"Failed to parse FASTA: {e}")
                return

        st.success(f"Parsed {len(records)} sequence(s). Running characterization models...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        profiles = []
        for i, rec in enumerate(records):
            status_text.text(f"Processing ({i+1}/{len(records)}): {rec['id']}...")
            prof = profile_multiple_sequences(
                records=[rec],
                organism_type=organism_type,
                tox_threshold=tox_thresh,
                alg_threshold=alg_thresh,
                ant_threshold=ant_thresh,
                candidate_type=candidate_type,
                rank_candidates=False,
                use_docker=use_docker,
                use_api=use_api,
            )[0]
            profiles.append(prof)
            progress_bar.progress((i + 1) / len(records))

        # Rank candidates
        profiles.sort(key=lambda x: x["desirability"]["score"], reverse=True)
        for rank, p in enumerate(profiles, start=1):
            p["desirability"]["rank"] = rank

        status_text.text("Profiling complete!")
        st.session_state["profiles"] = profiles

    # Results view
    if "profiles" in st.session_state and st.session_state["profiles"]:
        profiles = st.session_state["profiles"]
        df = to_dataframe(profiles)

        st.markdown("---")
        st.subheader("📊 Candidate Overview & Metrics")

        # KPI metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">Total Candidates</div>
                <div class="metric-val">{len(profiles)}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            antigens = sum(1 for p in profiles if p["antigenicity"]["is_antigen"])
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">Antigenic</div>
                <div class="metric-val" style="color: #3fb950;">{antigens}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            toxins = sum(1 for p in profiles if p["toxicity"]["is_toxic"])
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">Non-Toxic</div>
                <div class="metric-val" style="color: #58a6ff;">{len(profiles) - toxins}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            allergens = sum(1 for p in profiles if p["allergenicity"]["is_allergen"])
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">Non-Allergenic</div>
                <div class="metric-val" style="color: #e3b341;">{len(profiles) - allergens}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            top_des = profiles[0]["desirability"]["score"]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">Top Desirability</div>
                <div class="metric-val" style="color: #a371f7;">{top_des:.3f}</div>
            </div>""", unsafe_allow_html=True)

        # Ranked Table
        st.subheader("🏆 Candidate Ranking Table")
        display_cols = [
            "Desirability_Rank", "ID", "Length", "Mol_Weight_Da",
            "Antigenicity_Score", "Is_Antigen",
            "Toxicity_Score", "Is_Toxic",
            "Allergenicity_Score", "Is_Allergen",
            "Desirability_Score", "Is_Stable", "Epitopes_Count"
        ]
        st.dataframe(df[display_cols], use_container_width=True)

        # Individual Sequence Inspector
        st.markdown("---")
        st.subheader("🔍 Deep Sequence Inspector")
        selected_id = st.selectbox("Select peptide to inspect:", [p["id"] for p in profiles])
        selected_prof = next(p for p in profiles if p["id"] == selected_id)

        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("##### Physicochemical Properties")
            p_data = selected_prof["physicochemical"]
            st.write(f"**Sequence:** `{selected_prof['sequence']}`")
            st.write(f"**Molecular Weight:** {p_data['mol_weight']} Da")
            st.write(f"**GRAVY (Hydropathicity):** {p_data['gravy']}")
            st.write(f"**Instability Index:** {p_data['instability_index']} ({'Stable' if p_data['is_stable'] else 'Unstable'})")
            st.write(f"**Theoretical pI:** {p_data['isoelectric_point']}")
            st.write(f"**Net Charge at pH 7.0:** {p_data['charge_at_pH7']}")
            st.write(f"**Aromaticity:** {p_data['aromaticity']}")
            st.write(f"**Secondary Structure:** Helix: {p_data['secondary_structure']['helix']}, Sheet: {p_data['secondary_structure']['sheet']}, Turn: {p_data['secondary_structure']['turn']}")

        with colB:
            st.markdown("##### Biological Classifications")
            ant = selected_prof["antigenicity"]
            tox = selected_prof["toxicity"]
            alg = selected_prof["allergenicity"]
            des = selected_prof["desirability"]

            st.write(f"**Antigenicity:** {ant['score']:.3f} • {'✅ Antigen' if ant['is_antigen'] else '❌ Non-antigen'} ({ant.get('method', '')})")
            st.write(f"**Toxicity:** {tox['score']:.3f} • {'⚠️ Toxic' if tox['is_toxic'] else '✅ Non-toxic'} ({tox.get('method', '')})")
            st.write(f"**Allergenicity:** {alg['score']:.3f} • {'⚠️ Allergen' if alg['is_allergen'] else '✅ Non-allergen'} ({alg.get('method', '')})")
            st.write(f"**Combined Desirability:** {des['score']:.4f} (Rank #{des.get('rank', '-')})")

            # Epitope details
            ep_list = selected_prof["epitopes"]["regions"]
            st.markdown(f"##### Predicted B-Cell Epitopes ({len(ep_list)})")
            if ep_list:
                ep_df = pd.DataFrame(ep_list)
                st.dataframe(ep_df, use_container_width=True)
            else:
                st.caption("No linear B-cell epitopes detected above threshold.")

        # Download reports
        st.markdown("---")
        st.subheader("📥 Export & Download Reports")
        dcol1, dcol2, dcol3 = st.columns(3)

        csv_data = df.to_csv(index=False).encode("utf-8")
        dcol1.download_button(
            label="📄 Download CSV Report",
            data=csv_data,
            file_name="peptide_profile_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        json_data = json.dumps(profiles, indent=2).encode("utf-8")
        dcol2.download_button(
            label="📦 Download JSON Data",
            data=json_data,
            file_name="peptide_profile_report.json",
            mime="application/json",
            use_container_width=True,
        )

        from src.aggregator import generate_html_report
        html_data = generate_html_report(profiles, df).encode("utf-8")
        dcol3.download_button(
            label="🌐 Download HTML Interactive Report",
            data=html_data,
            file_name="peptide_profile_report.html",
            mime="text/html",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
