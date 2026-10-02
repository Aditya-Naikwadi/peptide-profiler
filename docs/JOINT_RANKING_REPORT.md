# Joint Multi-Objective Candidate Ranking & Cross-Axis Evaluation Report

> **Evaluation Protocol Compliance:** Evaluated on Outer Fold 0 held-out homology clusters (zero leakage).
> Multi-objective optimization performed over calibrated probabilities for Antigenicity, Toxicity, and Allergenicity.

## 1. Executive Summary & Gating Funnel

- **Screening Dataset:** 103 cluster-held-out test sequences.
- **Strict Gating Hierarchy Funnel:**
  1. **Abstain / OOD:** **0** sequences excluded immediately from ranking.
  2. **WHO/FAO Allergenicity:** High-risk regulatory flags attached to sequences exceeding 35% identity over 80 aa.
  3. **Hard Toxicity Cap:** **56** sequences excluded for safety violations.
  4. **Stability & Aggregation Soft Gate:**
     - **23** sequences flagged `unstable` (Instability index $\ge 40.0$).
     - **10** sequences flagged `aggregation_prone` (Aggregation score $> 0.20$ or high sheet fraction).
     - **0** sequences flagged `advisory_short_peptide` (Instability index noted as advisory for length $< 20$ aa).
- **Ranked Survivors:** **47** candidates advanced into multi-objective Pareto sorting.

---

## 2. Multi-Objective Pareto Sorting (Fronts & Crowding Distance)

Candidates are ranked using **Fast Non-Dominated Sorting (Deb et al., NSGA-II)** across:
$$\max \; z = [p_{\text{antigen}}, \; 1 - p_{\text{toxic}}, \; 1 - p_{\text{allergen}}]$$

- **Uncertainty-Aware Dominance:** Candidate $A$ dominates $B$ only if the advantage exceeds the statistical confidence margin ($\epsilon = 0.02$).
- **Diversity Tie-Breaking:** Within each front, candidates are sorted by descending **crowding distance**, ensuring representation across the entire trade-off spectrum.
- **Monotone Invariance:** Mathematically invariant to any strictly increasing rescaling of the axes (e.g., log, power, affine).

### Pareto Front Distribution

| Pareto Front Index | Candidate Count | Dominance Semantics |
| :---: | :---: | :--- |
| **Front 0** | **20** | Non-dominated Pareto Optimal Leads |
| **Front 1** | **13** | Dominated by Front 0 |
| **Front 2** | **5** | Dominated by Front 1 |
| **Front 3** | **6** | Dominated by Front 2 |
| **Front 4** | **3** | Dominated by Front 3 |

---

## 3. Cross-Axis Correlation & Trade-Off Analysis

> **Why Joint Profiling is Necessary:** In independent screening pipelines, candidates selected solely for high antigenicity frequently carry severe hidden toxicity or allergenicity liabilities.

### Empirical Correlation Matrix (Cluster-Held-Out Test Partition)

| Axis Pair | Pearson $r$ ($p$-value) | Spearman $\rho$ ($p$-value) | Biological Implication |
| :--- | :---: | :---: | :--- |
| **Antigenicity vs. Toxicity** | **0.227** ($p=0.0210$) | **0.321** | Weak/moderate trade-off; highly immunogenic bacterial motifs frequently co-occur with pore-forming cytotoxic mechanisms. |
| **Antigenicity vs. Allergenicity** | **0.767** ($p=0.0000$) | **0.775** | Cross-reactivity risk; candidates triggering immune recognition can stimulate IgE-mediated allergic responses. |
| **Toxicity vs. Allergenicity** | **0.280** ($p=0.0042$) | **0.338** | Distinct mechanisms; low correlation confirms independent biophysical modes of action. |

### Conflict & Overlap Statistics
- **High Antigenicity ($p \ge 0.50$):** 0 candidates.
- **Antigenic AND Toxic ($p_{\text{tox}} \ge 0.40$):** **0** candidates (0.0% of antigenic peptides).
- **Antigenic AND Allergenic ($p_{\text{alg}} \ge 0.50$):** **0** candidates (0.0% of antigenic peptides).
- **Ideal Multi-Objective Compromises (High Ant, Low Tox, Low Alg):** **0** candidates (0.0% of library).

---

## 4. Top 10 Pareto-Ranked Candidate Leads

| Rank | Candidate ID | Length | Pareto Front | Crowding Dist | $p_{\text{ant}}$ | $p_{\text{tox}}$ | $p_{\text{alg}}$ | Derringer-Suich Score | Reason Codes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | `P05059` | 449 aa | Front 0 | inf | 0.100 | 0.093 | 0.210 | 0.0 | unstable, extreme_net_charge |
| **#2** | `Q9SPL3` | 625 aa | Front 0 | inf | 0.080 | 0.158 | 0.100 | 0.0 | unstable |
| **#3** | `Q96DR5` | 249 aa | Front 0 | inf | 0.480 | 0.309 | 0.710 | 0.0 | SAFETY_VIOLATION_ALLERGENICITY, aggregation_prone, extreme_net_charge |
| **#4** | `P80384` | 141 aa | Front 0 | inf | 0.640 | 0.343 | 0.640 | 0.0 | aggregation_prone |
| **#5** | `P61513` | 92 aa | Front 0 | inf | 0.450 | 0.453 | 0.410 | 0.3597 | extreme_net_charge |
| **#6** | `P62241` | 208 aa | Front 0 | 0.4213 | 0.270 | 0.395 | 0.150 | 0.3505 | unstable, extreme_net_charge |
| **#7** | `P47914` | 159 aa | Front 0 | 0.4128 | 0.337 | 0.226 | 0.310 | 0.4627 | extreme_net_charge |
| **#8** | `P32969` | 192 aa | Front 0 | 0.371 | 0.210 | 0.215 | 0.270 | 0.2038 | extreme_net_charge |
| **#9** | `P63244` | 317 aa | Front 0 | 0.3445 | 0.470 | 0.347 | 0.590 | 0.1658 | PASS |
| **#10** | `O15240` | 615 aa | Front 0 | 0.3315 | 0.170 | 0.095 | 0.470 | 0.0 | unstable, extreme_net_charge |

---

## 5. Secondary Scalarization: Derringer-Suich Desirability

As a secondary scalarization view, each candidate is evaluated with the **Derringer-Suich (1980)** geometric mean desirability function:
$$D = (d_{\text{ant}}^{w_1} \cdot d_{\text{tox}}^{w_2} \cdot d_{\text{alg}}^{w_3})^{1 / \sum w}$$

- **Labeling Contract:** Clearly labeled as a scalarization (`"is_scalarization": true`).
- **Zero-Tolerance Property:** If any individual response breaches acceptable safety boundaries ($d_k = 0$), composite desirability strictly collapses to $0.0$.

---
*Report generated automatically under WP9.5.*
