# Proposed central location for local MC/data paths.
#
# NOT wired into any notebook yet -- this is a template. The 7 pipeline
# notebooks currently each define these paths inline (see README.md,
# "Paths you must edit"), with values suited to running from the repo
# root on the gpvm/local layout used to produce this archive:
#
#   .../nuNCQE_XS_analysis/v2.0/NCQE_XS_analysis/   <- repo root (this file's grandparent)
#   .../nuNCQE_XS_analysis/v2.0/MC_GENIE-O16/       <- one level up
#   .../nuNCQE_XS_analysis/FY22_23/                 <- two levels up
#   .../nuNCQE_XS_analysis/off_beam/                <- two levels up
#   .../nuNCQE_XS_analysis/skyshine_100k_no_ext_clusters*.root  <- two levels up
#
# To adopt: rename this file to lib/paths.py, point the constants at your
# own staged copies, then in each notebook replace the inline assignment
# with an import, e.g.:
#   from lib.paths import MC_DIRECTORY as MC_directory

MC_DIRECTORY = '../MC_GENIE-O16/'        # local copy of gpvm MC ntuples (or 'MC_GENIE-Base/')

DATA_DIRECTORY = '../../FY22_23/'
# on-beam data source: /pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_BEAM_DATA/

OFFBEAM_DIRECTORY = '../../off_beam/'
# off-beam data source: /pnfs/annie/persistent/users/doran/datasets/NCQE_BEAMCLUSTER_DATA/EXTRACTED_FULL_OFFBEAM_DATA/

SKYSHINE_FILES = [
    '../../skyshine_100k_no_ext_clusters.root',
    '../../skyshine_100k_no_ext_clusters_2.root',
    '../../skyshine_100k_no_ext_clusters_3.root',
]

# Aggregate MC ntuple (with follower gamma-ray info) used for the primary
# de-excitation systematic study. Any aggregate MC sample with follower
# information will do.
DEXCITATION_MC_FILE = '../GENIE-Empirical-4.417e20-ntuple_dirt_downsampled.root'
