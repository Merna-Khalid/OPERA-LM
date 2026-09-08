# Garden-path stimulus sources

All files downloaded 2026-09-07. Nothing here was hand-created; the two
`*_GPE_effects_*.csv` files were exported verbatim from the upstream `.rds`
files noted below.

## 1. SAP Benchmark (caplabnyu/sapbenchmark)

Repo: https://github.com/caplabnyu/sapbenchmark
Commit: 15e61066d510b5349e17740e6488c976abc3e1ac (2024-09-18, HEAD of `main`)
License: MIT (see `sapbenchmark_LICENSE`)
Reference (per repo readme): Huang, K.J., Arehalli, S., Kugemoto, M., Muxica, C.,
Prasad, G., Linzen, T., & Dillon, B. "A large-scale investigation of syntactic
processing reveals misalignments between humans and neural language models."
The ClassicGP subset stimuli are the van Schijndel & Linzen (2021, Cognitive
Science) garden-path items (24 items x NP/S, NP/Z, MV/RR, ambig/unambig pairs).

- `sapbenchmark_items_ClassicGP.csv` <- Surprisals/data/items_ClassicGP.csv
  Wide format, 72 rows (24 items x 3 constructions). Columns: item, condition
  (NPS/NPZ/MVRR x AMB/UAMB), disambPositionUnamb, unambiguous (sentence),
  disambPositionAmb, ambiguous (sentence), Question, Option1, Option0, Answer,
  "Ambiguity targeted?". Disambiguating-word positions are 1-indexed word
  counts into the respective sentence variant.
- `sapbenchmark_items_ClassicGP.pivot.csv` <- Surprisals/data/items_ClassicGP.pivot.csv
  Long format, 144 rows (one per item x construction x ambiguity). Adds
  `Sentence` (the actual string for that condition) and `disambPosition_0idx`
  (0-indexed position of the disambiguating word in that sentence).
- `sapbenchmark_items_filler.csv`, `sapbenchmark_items_filler.pivot.csv`
  <- Surprisals/data/items_filler.csv / .pivot.csv. 39 filler sentences drawn
  from the Provo corpus (`item#_in_Provo`), with comprehension questions.
- `sapbenchmark_Items_for_all_subsets.xlsx` <- "Items for all subsets.xlsx"
  Master item sheet for all SAP subsets; sheet `NPSNPZMVRR` = ClassicGP items
  (the unambiguous-sentence column is unnamed in this sheet).
- `sapbenchmark_readme_SAP.txt` <- readme_SAP.txt (provenance + data links)
- `sapbenchmark_LICENSE` <- LICENSE (MIT)
- `sapbenchmark_ClassicGP_by_item.rds` <- plots/spr/ClassicGP/by_item.rds
- `sapbenchmark_ClassicGP_by_construction.rds` <- plots/spr/ClassicGP/by_construction.rds
  EMPIRICAL human garden-path effects (GPE = ambiguous minus unambiguous RT,
  ms, posterior mean + 95% CrI) from the SAP self-paced-reading recollection
  (N=2000), per item (24 x 3 constructions) and per construction, at 3 regions
  of interest: ROI 0 = Critical (disambiguating word), 1 = Critical+1,
  2 = Critical+2 (mapping per analysis/shared/util.R lines 77, 91, 105).
- `sapbenchmark_ClassicGP_GPE_effects_by_item.csv`,
  `sapbenchmark_ClassicGP_GPE_effects_by_construction.csv`
  CSV exports of the two .rds files above (columns: item/ROI, coef=GPE_<constr>,
  mean, lower, upper, region).
- `sapbenchmark_util.R` <- analysis/shared/util.R (documents ROI coding and
  the `load_data()` function; raw per-participant RT files
  `ClassicGardenPathSet.csv` / `Fillers.csv` are NOT in the repo -- they are on
  Google Drive, links in readme_SAP.txt)
- `sapbenchmark_generate_plots.Rmd` <- plots/spr/generate_plots.Rmd

## 2. Arehalli, Dillon & Linzen (2022), arXiv:2210.12187

Repo: https://github.com/SArehalli/SyntacticSurprisal
Commit: 78910e5edecba1454c3acb052a03a73ae8c4c1fe (2025-10-25, HEAD of `main`)
License: none stated in the repo.
Uses the same ClassicGP items via the SAP Benchmark. Note: its
`items_ClassicGP.pivot.csv` has a labeling quirk -- the `condition` column is
`*_UAMB` on every row; the ambig/unambig distinction is carried by the
`ambiguity` column. Sentence set is identical to the SAP file (144 sentences).

- `arehalli2022_items_ClassicGP.pivot.csv` <- data/items_ClassicGP.pivot.csv
- `arehalli2022_items_filler.pivot.csv` <- data/items_filler.pivot.csv
- `arehalli2022_analysis.R` <- analysis/analysis.R (reads human SPR data from
  `ClassicGardenPathSet.csv` / `Fillers.csv`, i.e. the SAP Google Drive files)
- `arehalli2022_pairwise_comps.log` <- analysis/pairwise_comps.log
  Model-PREDICTED RT contrasts (lexical/syntactic/both/no-surprisal), NOT
  human data.
- `arehalli2022_README.md` <- README.md

## 3. van Schijndel & Linzen (2021), Cognitive Science 45(6):e12988
   "Single-stage prediction models do not explain the magnitude of syntactic
   disambiguation difficulty"

Repo: https://github.com/vansky/replications
Commit: 74ca78b108fff7a6bef95f1d957f3479ba0f76bf (2022-09-27, HEAD of `master`)
License: none stated in the repo.
Directory: vanschijndel_linzen-2021-cognitive_science/

- `vanschijndel_linzen2021_linking_function_regression.Rmd`
  The paper's analysis. Reads human SPR data from `./Data/spr_unmodified_combined.csv`,
  which it says comes from the Prasad & Linzen (2019) OSF page
  https://osf.io/57ckx/ -- but that OSF node publicly exposes only a poster PDF
  (checked 2026-09-07); the raw CSVs are not posted there or in this repo.
- `vanschijndel_linzen2021_listA_filler_sentences.output.rolled`
  Model surprisal/entropy output for the experiment's filler list (listA).
- `vansky_replications_README.md` <- README.md (paper-to-directory mapping)

## Notes

- The Cognitive Science 2021 paper by van Schijndel & Linzen is "Single-stage
  prediction models do not explain the magnitude of syntactic disambiguation
  difficulty" (e12988); "A Neural Model of Adaptation in Reading" is van
  Schijndel & Linzen, EMNLP 2018 (replication dir `vanschijndel_linzen-2018-emnlp`
  in the same repo; it contains dative-adaptation materials, no garden-path
  stimulus files).
- Natural Stories RT data were intentionally NOT fetched (not needed).
