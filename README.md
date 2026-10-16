# NCQE_XS_analysis (README powered by Claude AI)

Analysis notebooks for extracting the NCQE (neutral-current quasi-elastic)
cross section using the ANNIE experiment. Inputs are
ANNIE `PhaseIITriggerTree`/`PhaseIIClusterTree`-style ROOT ntuples (beam
data, off-beam data, and GENIE MC) staged locally by the user — none of
those files are included in this repository (see "Paths you must edit"
below).

## v3: adopted extraction (2026-10-15, branch `adopted-extraction`)

This copy reproduces the extraction and the uncertainty budget of the PRD draft. Notebooks 01-05 are unchanged from
v2; 06 and 07 are modified, and `08_systematics.py` is new.

**What changed in the physics (v2 → v3).**

| | v2 | v3 |
|---|---|---|
| OOFV νNCQE (selected νNCQE with a true vertex outside the FV) | subtracted as background | scaled with *f* together with the FV signal (T2K, PRD 100, 112009, Eq. 5) |
| Beam-uncorrelated (CIT) background | off-beam triggers, 127.44 (no FMV/MRD readout, so no veto) | pre-beam window [0,150) ns of the beam triggers: 67 × 182/150 = 81.29 (as in 03) |
| Skyshine | 14.1 ± 13.0 (04) | refit with the pre-beam CIT: 13.95 ± 12.87 |
| ⟨σ_GENIE⟩ | spline average 2.04334 | MC-implied *N*_true/(POT·Φ·*N*_T) = 2.0513 |
| Statistical uncertainty | √N/N | √N_obs/(M_sig + M_OOFVν), T2K convention |
| Interaction model / flux / MC stat. | 24.6 / 24.8 / 6.7 % (xsec_analyzer fake-data mode, OOFV νNCQE as background) | 8.6 / 10.0 / 5.2 % (paper extraction) |
| Primary γ, *p* / *s* shell | +17.3/−5.0 % / +2.5/−2.6 % (`formula_v2`, OOFV νNCQE as background) | +10.9/−3.2 % / +1.6/−1.6 % (paper extraction) |
| T2K-like "other" states | — | −15.2 % (one-sided) |

The extraction is in `lib/extraction.py` (paper Eqs. 25–28) and is shared by 06, 07 and 08.

**Run order.**

```
python3 08_systematics.py      # ~7 min cold (14 workers), ~10 s with cache/; writes results/08_systematics.json + log
jupyter nbconvert --to notebook --execute --inplace 06_Primary_Dexcitation_Systematic.ipynb   # results/06_primary_gamma.json
jupyter nbconvert --to notebook --execute --inplace 07_NCQE_XS_Extraction_Plotting.ipynb      # results/07_budget.json, plots/NCQE_XS.png
python3 09_xs_money_plot.py --out plots/NCQE_comparison_median_column.png   # paper figure (+ _values.json sidecar)
python3 10_t2k_comparison.py   # separation from T2K: four statistical treatments + flux / NCQE-like / de-excitation variants
```

Numbers are no longer copied by hand between notebooks; 06 and 07 read the JSON files in `results/`, each of which
records its provenance (date, git commit, software versions, input directories). Terms analysed elsewhere (secondary
γ rays, PMT tilt, energy scale, vertex, POT) are entered by hand in 07 and flagged there.

**New files.**

| File | Purpose |
|---|---|
| `08_systematics.py` | Stage A: extraction inputs not provided by 05 (pre-beam CIT, OOFV split, true FV νNCQE, base GENIE). Stage B: skyshine refit with the pre-beam CIT and the joint (f, α) fit. Stage C: interaction model (44-knob multisim), flux, GENIE unisims, MC statistics. Stage D: bunch timing. Stage E: CIT statistics, skyshine, External. |
| `09_xs_money_plot.py` | Paper cross-section figure (ANNIE vs GENIE, NEUT and T2K). It reads `inputs/xs_predictions.json`, `inputs/t2kflux_2020/` and the 07 budget. `--rebuild-inputs` re-derives `inputs/xs_predictions.json` from the generator outputs in `BuildEventGenerators/` (≈43 GB, not in this repository; md5 checksums are recorded). |
| `10_t2k_comparison.py` | Separation between ANNIE and T2K (PRD 100, 112009). Statistical treatments: the paper convention, symmetrised errors, split-normal Monte Carlo, and Barlow variable-variance likelihood. Physics variants: the flux-averaging correction, the NCQE-like signal definition, and T2K-like de-excitation. Output: `results/10_t2k_comparison.json`. |
| `inputs/` | Small tracked inputs: the reduced GENIE/NEUT predictions and FV flux (`xs_predictions.json`), and the T2K 2020/2013 public flux tables (see `SOURCE.txt`). |
| `lib/flux_energy.py` | Flux median / percentiles / mean, one algorithm for BNB and T2K (point placement in 09). |
| `lib/paths.py` | Input locations, relative to the `~/hep/ANNIE` layout; each can be overridden by an environment variable. |
| `lib/selection.py` | The 05 selection, vectorised; reproduces the 05 yields exactly. |
| `lib/extraction.py` | Paper Eq. 26 and ⟨σ_GENIE⟩; the v2 formula, kept for cross-checks only; provenance helper. |
| `lib/reweight.py` | Pass over the 44-knob reweighting production. |
| `lib/followers.py` | Follower-particle records for 06. |
| `cache/` | Derived arrays, gitignored, safe to delete. |
| `results/` | JSON hand-off between 06/08 and 07, plus logs. |

**Inputs (defaults in `lib/paths.py`).**
- `../NCQE-GENIE-samples/MC_GENIE-O16/` and `MC_GENIE-Base/`
- `../NCQE-Data/FY22_23/`
- `../NCQE_Paper_Box/xsec_analyzer_check/empirical_all_weights_no_downscaling/`, the 44-knob production. It holds the
  same events as `MC_GENIE-O16`, with re-randomised spill timing; 2 of its 3884 files are empty.

**Validation (2026-10-15, this branch).** Every stage of 08 prints its reference value next to the recomputed one
(`results/08_systematics_log.txt`). A rerun from `cache/` is bit-identical; the bootstraps are seeded.

| Quantity | Reproduced | Reference (paper / review) |
|---|---|---|
| CV selection | identical event by event to the frozen 05 selection | — |
| Yields | 152.61 / 140.38 / 47.76 / 7.00 / 2.91 / 113.01 / 78.23 | paper Table V |
| Data | N_obs 772, pre-beam 67 | 772, 67 |
| f, ⟨σ_GENIE⟩, σ | f = 1.4603, ⟨σ_GENIE⟩ = 2.0513, σ = 2.9955 ± 0.1945 | 1.46, 2.05, 3.00 ± 0.19 |
| Skyshine | 13.95 ± 12.87 | 13.95 ± 12.87 |
| Joint fit | α = 0.347 ± 0.091, f = 1.384 | 0.347, 1.384 |
| Interaction model / flux / MC stat. | 8.60 / 9.95 / 5.18 % | 8.6 / 10.0 / 5.2 % |
| Same, event-level cross-check | 8.29 / 9.43 % | — |
| Bunch timing | +12.32 / −7.39 % (narrowing −3.09 %, mirror −1.60 %) | +12.3 / −6.9 % in draft 1 (mirror) |
| External / skyshine / CIT stat. | 5.23 / 3.01 / 2.32 % | 5.2 / 3.0 / 2.3 % |
| Primary γ, *p* shell (06) | +10.90 / −3.16 %; v2-configuration cross-check +17.25 / −5.01 % | draft 1: +17.3 / −5.0 % |
| "Other" states (06) | −15.23 % (±2.0 % MC stat.) | −15.3 % |
| Base GENIE de-excitation | σ = 2.2845 (−23.7 %) | 2.28 (−24 %) |
| Figure (09), closure | the PRD draft-1 figure (`NCQE_comparison_median_systv4_column.png`) is reproduced pixel for pixel, with all 24 plotted values identical, when 09 is given that budget (`final_uncertainties_v4.json`) | — |
| Budget (07) | syst +24.65 / −25.17 %, total +25.49 / −26.00 %; σ = 3.00 ± 0.19 +0.74/−0.75 | draft 1: +28.1 / −25.5 % |


**Authorship of v3 code.** The v3 library modules, `08_systematics.py` and the v3 cells of 06 and 07 were written with
Claude (Anthropic), ported from the 2026-10 review scripts in `NCQE_Paper_Box/Working_Directory_Agent/`, under the
author's direction. The author's own classification code in 06 is unchanged.

## Pipeline / run order

Run notebooks 1-7 in order. Each is self-contained (imports from `lib/`,
scans the raw ntuples directly) but later steps reuse numbers that
earlier steps print out, copied in by hand — see each notebook's final
cell(s) for the values to carry forward. The two additional (unnumbered)
scripts below are appendix/QC utilities you can run any time afterward.

| # | Notebook | What it does |
|---|---|---|
| 1 | [`01_Bunch_Fitting_Data.ipynb`](01_Bunch_Fitting_Data.ipynb) | Fits bunch-timing centers/widths from on-beam **data** CC-inclusive events, for the NC/CC bunch-structure mapping used to boost NC purity. |
| 2 | [`02_Bunch_Fitting_MC.ipynb`](02_Bunch_Fitting_MC.ipynb) | Same bunch-timing fit, run on **MC** (CC-tagged sample), to compare/derive the MC-side bunch structure. |
| 3 | [`03_CCinc_Sideband.ipynb`](03_CCinc_Sideband.ipynb) | Dirt (out-of-tank) sideband using CC muons; derives the dirt-background scale factor (`ALPHA_SCALING`, currently 0.27 in `lib/constants.py`). |
| 4 | [`04_Skyshine_Sideband.ipynb`](04_Skyshine_Sideband.ipynb) | Sideband analysis of the "skyshine" neutron background, using the >2-sigma bunch-timing region; derives the expected skyshine rate in the signal region. |
| 5 | [`05_FULL_DATA_signal_band.ipynb`](05_FULL_DATA_signal_band.ipynb) | Main signal-region analysis: reconstructs data and MC (by category) event rates inside the 1-sigma bunch window across the full data/MC statistics; exports `Base_GENIE.root` / `Empirical_GENIE.root`. |
| 6 | [`06_Primary_Dexcitation_Systematic.ipynb`](06_Primary_Dexcitation_Systematic.ipynb) | Systematic study of the primary de-excitation gamma-ray uncertainty — these gammas are the NCQE signal itself. |
| 7 | [`07_NCQE_XS_Extraction_Plotting.ipynb`](07_NCQE_XS_Extraction_Plotting.ipynb) | Combines the counts/rates from steps 3-6 (entered by hand, see its "INPUTS" cell) into the final 1D NCQE cross section, the full systematic uncertainty budget (flux, GENIE xsec, MC stat, skyshine, dirt, primary/secondary de-excitation, DV, POT), and the official result plot (`plots/NCQE_XS.png`), for use alongside `xsec_analyzer`. |
| (extra) | [`Waterfall_Plot.ipynb`](Waterfall_Plot.ipynb) | Re-derives event counts at every selection stage (clusterization through the final 1/3-spill cut) directly from the raw MC/data/off-beam/skyshine ntuples, and assembles the event-selection waterfall table (efficiency, MC purity, data purity per stage) plus its plot, for Base and Empirical GENIE. |
| (extra) | [`plot_base_vs_empirical.py`](plot_base_vs_empirical.py) | Additional script to plot the PMT observables for both Base + Empirical model GENIE using the exported root files.|

## Repository layout

```
NCQE_XS_analysis/
├── 01_Bunch_Fitting_Data.ipynb
├── 02_Bunch_Fitting_MC.ipynb
├── 03_CCinc_Sideband.ipynb
├── 04_Skyshine_Sideband.ipynb
├── 05_FULL_DATA_signal_band.ipynb
├── 06_Primary_Dexcitation_Systematic.ipynb
├── 07_NCQE_XS_Extraction_Plotting.ipynb
├── Base_GENIE.root            # exported by 05, Base GENIE model event branches
├── Empirical_GENIE.root       # exported by 05, Empirical GENIE model event branches
├── (extra) Waterfall_Plot.ipynb        # event-selection waterfall table/plot
├── (extra) plot_base_vs_empirical.py   # uses the exported root files
├── lib/
│   ├── constants.py           # tank/FV geometry, bunch-timing constants, derived scale factors
│   ├── imports.py             # shared package imports + matplotlib style
│   ├── reco_functions.py      # reco energy, bunch cutting, direction/FV helper functions
│   ├── paths_template.py      # unused proposal for centralizing the local data paths below, not wired into any script
│   ├── POT_trigger_FY22_23_summary.csv   # per-run trigger counts / POT for on-beam data
│   └── offbeam_summary.csv               # per-run trigger counts for off-beam data
└── plots/                      # output directories only -- see note below
    ├── bunch_timing/
    ├── bunch_timing/MC/
    ├── sidebands/
    └── comparison/             # written by plot_base_vs_empirical.py
```

All notebooks assume the default Jupyter behavior of running with the
notebook's own directory as the kernel's working directory — since every
path below is written relative to the repo root with no `../`, **keep the
notebooks directly in the repo root** (don't move them into subfolders
without also updating those paths).

**Plots:** figures generated by these notebooks are not included in this
repository — they haven't been blessed by the collaboration yet. The same
applies to the derived table exports from `Waterfall_Plot.ipynb`
(`NCQE_waterfall_counts*.csv`, `NCQE_waterfall_table*.tex`) — excluded for
now alongside the images. The `plots/` subdirectories above are kept as
empty placeholders (via `.gitkeep`) so the expected output layout is
visible; each notebook recreates its own output directory at runtime
(`os.system('mkdir -p ' + plots_dir)` / `os.makedirs`), so nothing needs
to be created by hand before running.

## Paths you must edit

None of the raw data/MC is in this repository. Every notebook below has a
cell (commented `# REQUIRES USER INPUT` where present) defining local
directory/file paths that need to point at your own staged copies before
running:

| Notebook | Variable | Current value | Points at (source) |
|---|---|---|---|
| `01_Bunch_Fitting_Data.ipynb` | `directory` | `'FY22_23/'` | On-beam data directory. **Note:** this is a bare, zero-`../` path, inconsistent with the `'../../FY22_23/'` used by steps 3-5 below for what should be the same data — left as-is since scripts weren't edited; update it to match your actual layout. |
| `01_Bunch_Fitting_Data.ipynb` | `csv_file` | `'lib/POT_trigger_FY22_23_summary.csv'` | Repo-internal, no edit needed. |
| `02_Bunch_Fitting_MC.ipynb` | `MC_directory` | `'../MC_GENIE-O16/'` | Local copy of GENIE MC ntuples (or `'MC_GENIE-Base/'`). |
| `03_CCinc_Sideband.ipynb` | `MC_directory` | `'../MC_GENIE-O16/'` | Same as above. |
| `03_CCinc_Sideband.ipynb` | `data_directory` | `'../../FY22_23/'` | On-beam data: `/pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_BEAM_DATA/` |
| `03_CCinc_Sideband.ipynb` | `offbeam_directory` | `'../../off_beam/'` | Off-beam data: `/pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_OFFBEAM_DATA/` |
| `04_Skyshine_Sideband.ipynb` | `MC_directory`, `data_directory`, `offbeam_directory` | same as step 3 | same as step 3 |
| `04_Skyshine_Sideband.ipynb` | `skyshine_file_names` | `['../../skyshine_100k_no_ext_clusters.root', ..._2.root, ..._3.root]` | Simulated skyshine MC samples. |
| `05_FULL_DATA_signal_band.ipynb` | `MC_directory`, `data_directory`, `offbeam_directory`, `skyshine_file_names` | same as steps 3-4 | same as steps 3-4 |
| `06_Primary_Dexcitation_Systematic.ipynb` | `file_name` | `'../GENIE-Empirical-4.417e20-ntuple_dirt_downsampled.root'` | Any aggregate MC sample with follower gamma-ray information (files processed through `xsec_analyzer` are easiest to pair cluster:event). |
| `Waterfall_Plot.ipynb` | `MC_directory`, `data_directory`, `offbeam_directory`, `skyshine_file_names` | same as steps 3-5 | same as steps 3-5. **Note:** `MC_directory`'s inline comment (`# or 'MC_GENIE-Base/'`) is stale — the value is already `'MC_GENIE-Base/'`, the comment should read `'MC_GENIE-O16/'` as the alternative. |

All `lib/*.csv` and `lib/*.py` references are repo-internal and need no
changes.

## Outputs

`05_FULL_DATA_signal_band.ipynb` exports `Base_GENIE.root` and
`Empirical_GENIE.root` — reco-level event branches (energy, hits,
vertex/CV position, bunch/delta timing, GENIE weights) per event category,
one file per GENIE tune, written via `uproot.recreate`. These are kept in
the repository as small (~330 KB) reference/comparison artifacts;
`plot_base_vs_empirical.py` reads both back in for the Base-vs-Empirical
comparison plots.

`07_NCQE_XS_Extraction_Plotting.ipynb` saves the final cross section result
plot to `plots/NCQE_XS.png`.


## AI Disclaimer

All script and analysis infrastructure is my own. Claude was used to organize the analysis scripts to aid in documentation / reproducibility.
