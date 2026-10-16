# Local data / MC locations for the v3 scripts (06, 07, 08_systematics.py).
#
# Defaults assume this repository sits next to the staged inputs, i.e. the ~/hep/ANNIE layout used on the Mac
# (== /workspace/hep/ANNIE in the HEP container):
#
#   ANNIE/NCQE_XS_analysis_v3/                       <- this repository
#   ANNIE/NCQE-GENIE-samples/MC_GENIE-O16/           <- Empirical-model CV MC (per-file ntuples)
#   ANNIE/NCQE-GENIE-samples/MC_GENIE-Base/          <- base GENIE MC
#   ANNIE/NCQE-Data/FY22_23/                         <- on-beam data
#   ANNIE/NCQE_Paper_Box/xsec_analyzer_check/empirical_all_weights_no_downscaling/
#                                                    <- 44-knob reweighting production (same GENIE/WCSim events as
#                                                       MC_GENIE-O16, re-randomised spill timing and reconstruction)
#
# Any entry can be overridden with an environment variable of the same name, e.g. on the gpvm:
#   export NCQE_DATA_DIR=/pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_BEAM_DATA/
# The notebooks 01-05 still define their own paths inline (see README); only the v3 scripts read this file.
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNIE = os.path.dirname(REPO)


def _p(name, default):
    return os.environ.get(name, default)


MC_EMPIRICAL_DIR = _p('NCQE_MC_EMPIRICAL_DIR', os.path.join(ANNIE, 'NCQE-GENIE-samples', 'MC_GENIE-O16') + '/')
MC_BASE_DIR = _p('NCQE_MC_BASE_DIR', os.path.join(ANNIE, 'NCQE-GENIE-samples', 'MC_GENIE-Base') + '/')
MC_44KNOB_DIR = _p('NCQE_MC_44KNOB_DIR', os.path.join(ANNIE, 'NCQE_Paper_Box', 'xsec_analyzer_check',
                                                      'empirical_all_weights_no_downscaling') + '/')
DATA_DIR = _p('NCQE_DATA_DIR', os.path.join(ANNIE, 'NCQE-Data', 'FY22_23') + '/')
POT_CSV = os.path.join(REPO, 'lib', 'POT_trigger_FY22_23_summary.csv')
# Generator outputs used by the cross-section figure (09): GENIE splines, FV flux, NEUT flat trees and energy scan.
# Only read when rebuilding inputs/xs_predictions.json (09_xs_money_plot.py --rebuild-inputs).
GENERATORS_DIR = _p('NCQE_GENERATORS_DIR', os.path.join(ANNIE, 'NCQE_XS_analysis', 'BuildEventGenerators') + '/')
INPUTS_DIR = os.path.join(REPO, 'inputs') + '/'     # small, tracked inputs (T2K flux tables, reduced predictions)

CACHE_DIR = _p('NCQE_CACHE_DIR', os.path.join(REPO, 'cache') + '/')       # derived arrays, safe to delete (gitignored)
RESULTS_DIR = _p('NCQE_RESULTS_DIR', os.path.join(REPO, 'results') + '/')  # JSON hand-off 06/08 -> 07, logs

N_WORKERS = int(_p('NCQE_N_WORKERS', '14'))
