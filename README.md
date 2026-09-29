# Peptide Profiler: Open-Source Peptide Characterization Pipeline
### *An open-source, local alternative to VaxiJen, ToxinPred2 & AllerTOP2*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()

---

## 1. Objective

**Peptide Profiler** is a local, high-throughput bioinformatics pipeline that takes protein/peptide sequences in FASTA format and outputs a consolidated **"Nature Profile"** covering:

1. **Antigenicity**: Determines if a sequence is likely to trigger a protective immune response (*VaxiJen's core capability & Vaxign-ML*).
2. **Allergenicity**: Assesses probability of eliciting an allergic reaction (*AllerTOP2 & AlgPred2*).
3. **Toxicity**: Predicts if a peptide is potentially toxic or hemolytic (*ToxinPred2*).
4. **Physicochemical Properties**: Computes composition, charge, hydropathicity (GRAVY), molecular weight, theoretical pI, aromaticity, stability index, secondary structure propensities, and 30 Pfeature property descriptors.
5. **Linear B-Cell Epitope Mapping**: Identifies continuous antigenic epitope regions across sequences using the Kolaskar-Tongaonkar antigenicity scale.
6. **Candidate Desirability Ranking**: Ranks candidates using multi-objective optimization for vaccine design (high antigenicity, low toxicity, low allergenicity) or peptide therapeutics (low immunogenicity, low toxicity, low allergenicity).

---

## 2. Architecture

```text
                  input.fasta
                      │
                      ▼
             [Sequence Parser] ── validates FASTA, cleans/sanitizes sequences
                      │
     ┌────────────────┼────────────────┬────────────────┐
     ▼                ▼                ▼                ▼
[Physicochemical] [Antigenicity]   [Toxicity]    [Allergenicity]
   Biopython         Vaxign-ML     ToxinPred2       AlgPred2 /
   ProtParam          Docker       ONNX Model        AllerTOP2
       +                or             +            API or Local
   Pfeature         VaxiJen ML      Hybrid         ACC/PCP Model
  (AAC,PCP,ACR)     Classifier     (blastp)
     │                │                │                │
     └────────────────┼────────────────┴────────────────┘
                      │
                      ▼
               [Aggregator] ── merges outputs + B-cell epitopes + desirability
                      │
                      ▼
       report.csv  /  report.json  /  report.html
```

---

## 3. Repository Structure

```text
peptide-profiler/
├── README.md
├── requirements.txt
├── Dockerfile                   # Full pipeline containerization (Stretch Goal 3)
├── docker-compose.yml           # Runs Streamlit dashboard + Vaxign-ML container
├── app.py                       # Interactive Streamlit dashboard
├── docker/
│   └── vaxign-ml/
│       ├── docker-compose.yml   # Standalone Vaxign-ML Docker compose
│       └── run_vaxign.sh        # Shell execution wrapper
├── models/
│   ├── toxinpred2_rf.onnx       # Official ToxinPred2 Random Forest model
│   ├── antigen_rf.joblib        # Calibrated VaxiJen-style ML classifier
│   └── allergen_rf.joblib       # Calibrated AllerTOP-style ACC classifier
├── src/
│   ├── __init__.py
│   ├── parser.py                # FASTA input handling & residue validation
│   ├── physicochem.py           # Pfeature (AAC, PCP, ACR) + ProtParam wrapper
│   ├── antigenicity.py          # Vaxign-ML Docker & VaxiJen ML classifier
│   ├── toxicity.py              # ToxinPred2 ONNX inference & hybrid wrapper
│   ├── allergenicity.py         # AlgPred2/AllerTOP2 API & local ACC classifier
│   ├── aggregator.py            # Result merging, candidate ranking & report builder
│   └── cli.py                   # Command-line entry point
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # 18 unit & integration tests
└── examples/
    ├── sample_peptides.fa       # Curated benchmark dataset
    ├── report.csv               # Example CSV output
    ├── report.json              # Example JSON output
    └── report.html              # Example interactive HTML report
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Optional: Docker (for running Vaxign-ML Docker container)
- Optional: NCBI BLAST+ (`blastp`) on PATH (only for ToxinPred2 hybrid sequence-similarity model)

### Quick Start with `uv` or `pip`
```bash
# Clone the repository
git clone https://github.com/your-username/peptide-profiler.git
cd peptide-profiler

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage

### Command-Line Interface (CLI)

Run characterization on a FASTA file:
```bash
python src/cli.py -i examples/sample_peptides.fa -o report.csv
```

#### Advanced CLI Options
```bash
# Export all report formats (CSV, JSON, and interactive HTML)
python src/cli.py -i input.fa -o results/report --format all

# Screen candidates for therapeutic drug design (minimizing immunogenicity)
python src/cli.py -i candidates.fa -o report.csv --candidate-type therapeutic

# Adjust classification cutoffs
python src/cli.py -i input.fa -o report.csv --tox-thresh 0.55 --ant-thresh 0.45 --alg-thresh 0.50

# Enable Vaxign-ML Docker container (requires Docker daemon)
python src/cli.py -i input.fa -o report.csv --use-docker --organism bacteria

# Automatically sanitize non-standard amino acid characters
python src/cli.py -i input.fa -o report.csv --sanitize
```

### Interactive Streamlit Dashboard

Launch the browser interface:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser. Features:
- Upload FASTA files or test sample benchmarks.
- Live progress bars and multi-module profiling.
- Interactive candidate ranking table with color badges.
- Deep sequence inspection (physicochemical properties, secondary structure, B-cell epitopes).
- One-click downloads for CSV, JSON, and standalone HTML reports.

### Docker & Docker Compose

Run the entire pipeline and dashboard in Docker:
```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8501`.

---

## 6. Python API & Module Interface

Every module adheres to the consistent `run(sequence: str) -> dict` interface:

### Physicochemical Module
```python
from src.physicochem import run_physicochemical

res = run_physicochemical("ALWKTLLKKVLKAAAKA")
# Returns:
# {
#   'mol_weight': 1853.34,
#   'gravy': 0.429,
#   'instability_index': 8.36,
#   'is_stable': True,
#   'isoelectric_point': 10.85,
#   'charge_at_pH7': 4.96,
#   'aromaticity': 0.059,
#   'secondary_structure': {'helix': 0.824, 'turn': 0.176, 'sheet': 0.0},
#   'aac': {'A': 29.412, 'C': 0.0, ...},
#   'pcp_vector': [0.294, 0.0, 0.706, ...],  # 30 dimensions
#   'acr_dict': {'ACR_lag_1': 1.42, ...}
# }
```

### Antigenicity Module
```python
from src.antigenicity import run_antigenicity

res = run_antigenicity("MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK", organism_type="bacteria")
# Returns:
# {'antigenicity_score': 0.79, 'is_antigen': True, 'method': 'VaxiJen-Alternative (ACC/PCP ML Classifier)'}
```

### Toxicity Module
```python
from src.toxicity import run_toxicity

res = run_toxicity("GIGAVLKVLTTGLPALISWIKRKRQQ")  # Melittin
# Returns:
# {'toxicity_score': 0.650, 'is_toxic': True, 'method': 'ToxinPred2 (ONNX RF Model)'}
```

### Allergenicity Module
```python
from src.allergenicity import run_allergenicity

res = run_allergenicity("GVFNYETETTSVIPAARLFKAFILDGDNLFPKVAPQAISSVENIEGNGGPGTIKKISFPEGFPFKYVKDRVDEVDHTNFKYNYSVIEGGPIGDTLEKISNEIKIVATPDGGSILKISNKYHTKGDHEVKAEQVKASKEMGETLLRAVESYLLAHSDAYN")  # Bet v 1
# Returns:
# {'allergenicity_score': 0.64, 'is_allergen': True, 'method': 'Local ACC-based Classifier (Pfeature descriptors)'}
```

---

## 7. Output Format (per sequence)

```json
{
  "id": "OspA_Bacterial_Antigen",
  "sequence": "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK",
  "antigenicity": {
    "score": 0.79,
    "is_antigen": true,
    "method": "VaxiJen-Alternative (ACC/PCP ML Classifier)"
  },
  "allergenicity": {
    "score": 0.73,
    "is_allergen": true,
    "method": "Local ACC-based Classifier (Pfeature descriptors)"
  },
  "toxicity": {
    "score": 0.488,
    "is_toxic": false,
    "method": "ToxinPred2 (ONNX RF Model)"
  },
  "physicochemical": {
    "mol_weight": 5938.91,
    "gravy": -0.677,
    "instability_index": 29.58,
    "is_stable": true,
    "isoelectric_point": 9.41,
    "charge_at_pH7": 3.87,
    "aromaticity": 0.0,
    "secondary_structure": {
      "helix": 0.357,
      "sheet": 0.286,
      "turn": 0.232
    }
  },
  "desirability": {
    "score": 0.1092,
    "rank": 2,
    "candidate_type": "vaccine"
  },
  "epitopes": {
    "count": 3,
    "regions": [
      {
        "start": 1,
        "end": 14,
        "length": 14,
        "sequence": "MKTLLILAVVAAAL",
        "mean_score": 1.258
      }
    ]
  }
}
```

---

## 8. Integration Smoke Test Controls (Sanity Checks)

> **MLOps Notice**: The sequences below are canonical biological reference standards present in upstream training sets. They serve as **deterministic integration smoke tests** to verify pipeline execution, **not** as statistical validation of model generalization on unseen proteomes.

| Sequence ID | Description | Length | Antigenicity (Score / Flag) | Toxicity (Score / Flag) | Allergenicity (Score / Flag) | Stability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OspA_Antigen** | Borrelia burgdorferi outer surface antigen | 56 | **0.79 (Antigen)** | 0.49 (Non-toxic) | 0.73 (Allergen) | Stable (29.58) |
| **Spike_RBD** | SARS-CoV-2 Spike RBD epitope | 208 | **0.41 (Score: 0.41)** | 0.56 (Non-toxic) | 0.69 (Allergen) | Stable (31.78) |
| **Melittin** | Apis mellifera bee venom toxin | 26 | 0.53 (Score: 0.53) | **0.65 (Toxic)** | 0.46 (Non-allergen) | Unstable (44.60) |
| **Conotoxin** | Conus geographus neurotoxin | 25 | 0.40 (Non-antigen) | **0.98 (Toxic)** | 0.37 (Non-allergen) | Unstable (83.85) |
| **Bet_v_1** | Betula verrucosa birch pollen major allergen | 159 | 0.23 (Non-antigen) | 0.26 (Non-toxic) | **0.64 (Allergen)** | Stable (23.94) |
| **Ara_h_1** | Peanut seed storage allergen fragment | 57 | 0.45 (Non-antigen) | 0.45 (Non-toxic) | **0.72 (Allergen)** | Unstable (62.33) |
| **Exendin_4** | GLP-1 receptor agonist therapeutic peptide | 39 | 0.27 (Non-antigen) | **0.35 (Non-toxic)** | 0.53 (Allergen) | Unstable (68.10) |
| **Ubiquitin** | Human intracellular housekeeping protein | 76 | 0.26 (Non-antigen) | **0.27 (Non-toxic)** | **0.26 (Non-allergen)**| Stable (15.54) |

---

## 9. Running Tests

Run all 18 automated tests covering all modules, ONNX models, and CLI:
```bash
pytest -v tests/test_pipeline.py
```

---

## 10. License & Citation

Distributed under the MIT License. If you use Peptide Profiler in your research, please cite:
- **Biopython**: Cock et al., *Bioinformatics*, 2009.
- **Pfeature**: Pande et al., *Briefings in Bioinformatics*, 2023.
- **ToxinPred2**: Sharma et al., *Briefings in Bioinformatics*, 2022.
- **VaxiJen**: Doytchinova & Flower, *BMC Bioinformatics*, 2007.
- **Vaxign-ML**: Ong et al., *Bioinformatics*, 2020.
