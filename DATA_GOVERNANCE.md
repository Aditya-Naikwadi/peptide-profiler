# DATA_GOVERNANCE: Sources, Licenses, Provenance, Curation

Rule: **no data is downloaded, used or committed until its row here is verified and the owner approves it (Gate 1).** "UNVERIFIED" means the agent must read the source's own terms page and record the result, or ask the owner. It does not mean the data is unusable, only that use is not yet approved.

## 1. Source registry
| Source | URL | Version / Date | License / Terms Stated | Redistribution Allowed? | Label Evidence Type | Status |
|---|---|---|---|---|---|---|
| **VaxiJen bacterial immunogens** (Zaharieva et al., 2019) | `http://ddg-pharmfac.net/vaxijen/dataset` | 2019 (317 bacterial immunogens) | None explicitly stated on download page; citation required | Unknown / Unverified | Curated / Experimental (human challenge) | **UNVERIFIED** (Requires owner approval) |
| **VaxiJen v3.0 bacterial negatives** (Dimitrov et al., 2020) | `http://ddg-pharmfac.net/vaxijen/dataset` | Vaccines 2020 (317 non-immunogens) | Paper is CC BY 4.0; raw web files have no explicit license | Unknown / Unverified | Curated (non-immunogen bacterial proteins) | **UNVERIFIED** (Requires owner approval) |
| **VaxiGen tumor database** (Doneva et al., 2021) | `http://ddg-pharmfac.net/vaxijen3/tumordb` | 2021 (546 proteins, 212 peptides) | **CC BY 4.0** (Open Access, Biomedicines 2021) | **Yes** (with attribution) | Curated / Experimental (human tumor antigens) | **VERIFIED** |
| **VaxiJen viral data** (Doneva & Dimitrov, 2024) | `http://ddg-pharmfac.net/vaxijen3/` | IJMS 2024, 25:2949 | Paper is CC BY 4.0; standalone dataset license unconfirmed | Unknown / Unverified | Curated (viral antigens & non-antigens) | **UNVERIFIED** (Requires owner approval) |
| **IEDB** (Immune Epitope Database) | `https://www.iedb.org/` | Current continuous release | Free public research access; commercial licensing required via LJI | **Research only** (Commercial restricted) | Experimental (in vitro/in vivo assay validated) | **VERIFIED (Research only)** |
| **Protegen** (VIOLIN Database) | `https://violinet.org/protegen/` | Current continuous release | Open access for research with citation; commercial licensing via VIOLIN | **Research only** (Commercial restricted) | Experimental (in vivo protective antigens) | **VERIFIED (Research only)** |
| **UniProt / Swiss-Prot** | `https://www.uniprot.org/` | Current release | **CC BY 4.0** (Creative Commons Attribution) | **Yes** (unrestricted with attribution) | Curated (reviewed Swiss-Prot literature annotations) | **VERIFIED** |
| **ToxinPred / ToxinPred2 datasets** | `https://webs.iiitd.edu.in/raghava/toxinpred2/` | Sharma et al., 2022 | Free academic/research use with citation; redistribution unconfirmed | Unknown / Unverified | Curated (Swiss-Prot KW-0800 toxins & non-secretory controls) | **UNVERIFIED** (Requires owner approval) |
| **AllergenOnline** (FARRP) | `http://www.allergenonline.org/` | Version 21 (Univ. of Nebraska) | Freely accessible for safety research; no explicit redistribution grant | Unknown / Unverified | Curated (peer-reviewed clinical IgE binding) | **UNVERIFIED** (Requires owner approval) |
| **COMPARE** (HESI) | `https://comparedatabase.org/` | Annual peer-reviewed release | Public collaborative database; publications under CC BY | **Yes** (Public scientific resource) | Curated / Peer-reviewed (clinical IgE binding) | **VERIFIED** |

*Rule:* Data marked **UNVERIFIED** or **Research only** may be used for local training only with owner approval, and must never be committed to git or redistributed. Only **VERIFIED** CC BY 4.0 data (e.g. UniProt, VaxiGen tumor, COMPARE) may be committed or redistributed.

## 2. Third-party server policy
- VaxiJen and AllerTOP outputs are **predictions, not ground truth**. Do not use them as training labels.
- Do not automate queries to these servers. The owner submits comparison FASTA files manually via the server's upload/batch page, within its terms, and returns a results CSV.
- Note: "2.1" in the literature usually refers to AllerTOP v2.1, not VaxiJen. The current documented VaxiJen version is v3.0.

## 3. Provenance schema (every sequence row)
`seq_id, sequence, source, accession, deposit_date, organism, label, label_evidence, license, ingest_date, transformations, removal_reason (if dropped)`

## 4. License gate
- Approved and license-verified: usable, committed only if redistribution is allowed.
- UNVERIFIED or restricted: local research use only with owner approval; never committed to git; never redistributed; excluded from any shared artifact.
- Model weights trained on restricted data: flag in the model card.

## 5. Curation rules
1. Sanitize to the 20 standard amino acids; log every altered residue; flag sequences above a configurable altered fraction (`OWNER-INPUT`, suggest starting point 5%).
2. Drop fragments per documented rules (minimum length `OWNER-INPUT`); keep short peptides only in peptide-specific sets.
3. Remove exact and near-duplicates; resolve conflicting labels by a documented rule; log every removal.
4. **Negatives:** sample from UniProt; length-match to positives; exclude toxin, allergen and known-antigen annotations; filter by homology against positives; build per organism; document residual label noise.
5. Keep dates for temporal splits; never mix a source's positives and negatives in a way that lets source identity predict the label (verify with the source-held-out test).

## 6. Data card template (one per dataset)
Name/version · sources and licenses · size and class balance · length histogram · date range · organism mix · known biases and label noise · intended use · not intended for · cluster version · hash of the curated file.
