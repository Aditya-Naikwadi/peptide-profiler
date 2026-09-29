# DATA_GOVERNANCE: Sources, Licenses, Provenance, Curation

Rule: **no data is downloaded, used or committed until its row here is verified and the owner approves it (Gate 1).** "UNVERIFIED" means the agent must read the source's own terms page and record the result, or ask the owner. It does not mean the data is unusable, only that use is not yet approved.

## 1. Source registry
| Source | Content | What is known | License / terms | Status |
|---|---|---|---|---|
| VaxiJen bacterial immunogen set (Zaharieva et al., 2019) | 317 experimentally supported bacterial immunogens (human-tested; ~47 species); positives for bacterial antigenicity | Described as freely downloadable at ddg-pharmfac.net/vaxijen/dataset | Not confirmed | UNVERIFIED |
| VaxiJen v3.0 bacterial negatives | 317 non-immunogens used to train v3.0 | Described in the v3.0 paper (Dimitrov et al., Vaccines 2020, 8:709) | Downloadability and license not confirmed | UNVERIFIED |
| VaxiGen tumor database | 546 immunogenic human proteins, 212 tumor peptides, plus non-immunogen sets | Excel downloads at ddg-pharmfac.net/vaxijen3/tumordb | CC BY 4.0 (attribution required) | Verify on page, then approve |
| VaxiJen viral data (Doneva & Dimitrov, IJMS 2024, 25:2949) | Viral immunogens/non-immunogens | Hosted on the VaxiJen 3 site | Not confirmed | UNVERIFIED |
| IEDB | Epitope/antigen assay data | Not yet checked | Check | UNVERIFIED |
| Protegen | Protective antigens | Not yet checked | Check | UNVERIFIED |
| UniProt | Sequences with deposit dates; negatives | Not yet checked | Check | UNVERIFIED |
| ToxinPred datasets | Toxic/non-toxic peptides | Not yet checked | Check | UNVERIFIED |
| AllergenOnline / COMPARE | Allergen references | Not yet checked | Check | UNVERIFIED |

Agent: for each row, record the URL you used, version/date, exact license text or "none stated", redistribution allowed (yes/no/unknown), and label evidence type (experimental / curated / predicted). Update this table and stop for approval.

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
