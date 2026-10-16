# Energy summaries of tabulated fluxes, one algorithm for the BNB (ANNIE FV) and T2K (SK, FHC) fluxes, used for the
# horizontal placement of the points in the cross-section figure (09). Ported unchanged from
# NCQE_Paper_Box/Working_Directory_Agent/xs_plot/flux_energy_summary.py (2026-10-09).
#
# T2K: official 2020 flux release (13av7.1), SK +250 kA, inputs/t2kflux_2020/ (see SOURCE.txt), per 50 MeV on
#      variable-width bins -> per-bin flux = value x width / 0.05 GeV; integrated to 10 GeV as in PRD 100, 112009.
# Quantities: mean; median and 16th/84th percentiles (cumulative flux interpolated linearly within each bin, i.e. flat
# density per bin); lower/upper semi-deviations about the mean.
# T2K conventions: 2014 (PRD 90, 072012): median 0.63 GeV, 16th-84th percentiles 0.40-0.95 GeV of the 2014 flux;
#                  2019 (PRD 100, 112009, Fig. 12): mean 0.82 GeV, bar 0.50-1.75 GeV.
import os
import numpy as np

from . import paths

T2K_DIR = os.path.join(paths.INPUTS_DIR, 't2kflux_2020')


def summary(edges, flux):
    """edges (n+1), flux per bin (n, already integrated over the bin)."""
    c = 0.5 * (edges[1:] + edges[:-1])
    m = (c * flux).sum() / flux.sum()
    cum = np.concatenate(([0.0], np.cumsum(flux))) / flux.sum()
    q16, q50, q84 = (np.interp(q, cum, edges) for q in (0.16, 0.50, 0.84))
    b, a = c < m, c >= m
    sl = np.sqrt((flux[b] * (c[b] - m) ** 2).sum() / flux[b].sum())
    su = np.sqrt((flux[a] * (c[a] - m) ** 2).sum() / flux[a].sum())
    return dict(mean=m, median=q50, q16=q16, q84=q84, semidev_lo=sl, semidev_hi=su, mean_minus=m - sl, mean_plus=m + su)


def read_t2k(name, emax=10.0):
    rows = [l.split() for l in open(os.path.join(T2K_DIR, name)) if l.strip() and l.split()[0].isdigit()]
    lo = np.array([float(r[1]) for r in rows]); hi = np.array([float(r[3]) for r in rows])
    numu = np.array([float(r[4]) for r in rows]); nue = np.array([float(r[6]) for r in rows])
    assert np.allclose(lo[1:], hi[:-1]), 'non-contiguous bins'
    k = hi <= emax + 1e-9
    w = (hi - lo) / 0.05                                         # per-50-MeV density -> per-bin flux
    edges = np.concatenate(([lo[0]], hi[k]))
    return edges, ((numu + nue) * w)[k], (numu * w)[k], (nue * w)[k]
