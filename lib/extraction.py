# Cross-section extraction used by the paper (v3, "adopted extraction"), shared by 06, 07 and 08_systematics.py.
#
#   N_obs = f (M_sig + M_OOFVnu) + M_bkg + D_CIT + D_sky                                   (paper Eq. 25)
#   f     = (N_obs - M_bkg - D_CIT - D_sky) / (M_sig + M_OOFVnu)                          (paper Eq. 26)
#   <sigma> = f <sigma_GENIE>,   <sigma_GENIE> = N_true / (POT Phi N_T)                   (MC-implied, paper Eq. 28)
#   stat:  delta f = sqrt(N_obs) / (M_sig + M_OOFVnu)                                     (T2K convention)
#
# M_sig      : selected nuNCQE on 16O with a true vertex in the FV
# M_OOFVnu   : selected nuNCQE on 16O with a true vertex in the tank but outside the FV (scaled with f, as in T2K)
# M_bkg      : NCother + CC + nubarNCQE + other OOFV + alpha_dirt x External
# D_CIT      : pre-beam estimate of the continuous-in-time background (03 convention, window [0, 150) ns)
# All MC yields are TunedCV-weighted and scaled to the data POT.
#
# The 'earlier' configuration (notebook 07 of v2: OOFV nuNCQE subtracted as background, off-beam CIT, skyshine 14.1)
# is kept only to reproduce the pre-2026-10-09 numbers in cross-checks; it is not used for any result.
import numpy as np

from .constants import ALPHA_SCALING

PHI_FV = 2.08722e-8          # nu/cm^2/POT, nu_mu + nu_e flux integrated over energy, averaged over the FV
                             # (ANNIE_FLUX_r_0.7m_y_1m.root; BuildEventGenerators/, not part of this repository)
N_TARGET = 1.028e29          # 16O nuclei in the FV (r < 0.7 m, |y| < 1.0 m, Gd-loaded water; thesis App. E)
SIGMA_GENIE_SPLINE = 2.04334 # 1e-38 cm^2: GENIE spline flux average (nu_mu + nu_e, to 10 GeV); cross-check only

# earlier configuration (v2 notebook 07), for cross-checks only
D_OFFBEAM_V2 = 127.43794062670655   # 52 off-beam events x trigger ratio 2.4507 (no FMV/MRD readout in off-beam triggers)
D_SKY_V2 = 14.1

BKG = ('NCother', 'CC', 'nubar', 'OOFV_other', 'Ext')


def yields(sel, f_pot, keep=None, w=None, alpha=ALPHA_SCALING):
    """POT-scaled yields per category. `keep` defaults to the CV selection (in_win & in26); `w` to the CV weights.
    `w` may be (n,) or (n, U) for U universes."""
    keep = (sel['in_win'] & sel['in26']) if keep is None else keep
    w = sel['w'] if w is None else w
    w = np.where(np.isfinite(w), w, 0.0)
    cat, onu = sel['cat'], sel['oofv_nu']
    masks = dict(sig=(cat == 0), OOFV_nu=onu, NCother=(cat == 1), CC=(cat == 2), nubar=(cat == 3),
                 OOFV_other=(cat == 4) & ~onu, Ext=(cat == 5))
    k = keep if w.ndim == 1 else keep[:, None]
    out = {}
    for c, m in masks.items():
        mm = m if w.ndim == 1 else m[:, None]
        out[c] = f_pot * (w * (k & mm)).sum(0) * (alpha if c == 'Ext' else 1.0)
    return out


def m_bkg(Y):
    return sum(Y[c] for c in BKG)


def extract_f(Y, n_obs, d_cit, d_sky):
    """Paper Eq. 26 (works elementwise on universe arrays)."""
    return (n_obs - m_bkg(Y) - d_cit - d_sky) / (Y['sig'] + Y['OOFV_nu'])


def extract_f_earlier(Y, n_obs, d_cit=D_OFFBEAM_V2, d_sky=D_SKY_V2):
    """v2 notebook 07: OOFV nuNCQE subtracted as background (cross-checks only)."""
    return (n_obs - m_bkg(Y) - Y['OOFV_nu'] - d_cit - d_sky) / Y['sig']


def sigma_genie(n_true_potscaled, pot_data_1e20):
    """MC-implied flux-averaged GENIE nuNCQE cross section [1e-38 cm^2 / oxygen]."""
    return n_true_potscaled / (pot_data_1e20 * 1e20 * PHI_FV * N_TARGET) / 1e-38


def stat_f(Y, n_obs):
    return np.sqrt(n_obs) / (Y['sig'] + Y['OOFV_nu'])


def quad(up, down):
    """Quadrature sums of lists of (positive) upward and downward percentages."""
    return float(np.sqrt(np.sum(np.square(up)))), float(np.sqrt(np.sum(np.square(down))))


def provenance(extra=None):
    """Software versions and repository state, attached to every JSON result."""
    import sys, subprocess, platform, datetime, os
    import numpy, uproot, awkward, scipy
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True).stdout.strip()
    p = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), python=sys.version.split()[0],
             platform=platform.platform(), numpy=numpy.__version__, uproot=uproot.__version__,
             awkward=awkward.__version__, scipy=scipy.__version__, git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
             git_commit=git('rev-parse', '--short', 'HEAD'), git_dirty=bool(git('status', '--porcelain')))
    if extra:
        p.update(extra)
    return p
