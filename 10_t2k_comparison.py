#!/usr/bin/env python
"""
10_t2k_comparison.py -- separation between the ANNIE result and the T2K nuNCQE measurement (PRD 100, 112009), v3.

Run after 07 (and 09 for inputs/xs_predictions.json):   python3 10_t2k_comparison.py
Output: results/10_t2k_comparison.json and a printed table.

ANNIE : sigma, total uncertainties from results/07_budget.json (asymmetric).
T2K   : 1.70 +- 0.17 (stat.) +0.51/-0.38 (syst.) -> total +0.538/-0.416 (stat. and syst. in quadrature).
The two measurements are treated as independent (different beam, detector, generator, de-excitation model).

Statistical treatments of the difference D = sigma_ANNIE - sigma_T2K (ANNIE is the higher value):
  (a) paper convention     : D / sqrt(sigma_ANNIE,down^2 + sigma_T2K,up^2)  (the two uncertainties that face each other)
  (b) symmetrised          : D / sqrt(<sigma_ANNIE>^2 + <sigma_T2K>^2), <> = mean of up and down
  (c) split normals (MC)   : each result as a two-piece Gaussian; p = P(D_MC <= 0) one-sided, converted to a two-sided z
  (d) Barlow likelihood    : variable-variance Gaussians (R. Barlow, physics/0406120), V(mu) = s+ s- + (s+ - s-)(mu - x);
                             z = sqrt(min_mu sum chi2_i(mu))  (1 dof, test of a common value)
Physics variants (each with treatment (a)):
  (e) flux correction      : the two results are averages over different fluxes (BNB median 0.70 GeV, T2K 0.61 GeV);
                             T2K is moved to the BNB flux with the model ratio <sigma>_BNB / <sigma>_T2K (GENIE spline;
                             NEUT energy scan as cross-check), and ANNIE / T2K are compared at the same flux
  (f) NCQE-like            : ANNIE with NC 2p2h counted as signal, as in T2K's NEUT-based definition (+7.7 %, from
                             NCQE_Paper_Box/Working_Directory_Agent/nc2p2h_check, same CV), same fractional uncertainties
  (g) T2K-like de-excitation: ANNIE with the "other" states emitting like the 1s1/2 hole (T2K's nominal; 06), with the
                             one-sided "others" term removed from the ANNIE uncertainty (it is the CV choice there)
"""
import os, json
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from lib import paths
from lib.flux_energy import read_t2k
from lib.extraction import provenance, quad

B = json.load(open(os.path.join(paths.RESULTS_DIR, '07_budget.json')))
R06 = json.load(open(os.path.join(paths.RESULTS_DIR, '06_primary_gamma.json')))
P = json.load(open(os.path.join(paths.INPUTS_DIR, 'xs_predictions.json')))
NCQE_LIKE_SHIFT = 0.077          # nc2p2h_check/ncmec_yields_log.txt: drop all NC MEC from M_bkg -> sigma 3.226 (+7.7 %)

T2K = dict(val=1.70, stat=0.17, syst_up=0.51, syst_down=0.38)
T2K['up'], T2K['down'] = np.hypot(T2K['stat'], T2K['syst_up']), np.hypot(T2K['stat'], T2K['syst_down'])
A = dict(val=B['sigma'], up=B['sigma'] * B['total'][0] / 100, down=B['sigma'] * B['total'][1] / 100)
RNG = np.random.default_rng(20261015)


def z_paper(a, t):
    return (a['val'] - t['val']) / np.hypot(a['down'], t['up'])


def z_sym(a, t):
    return (a['val'] - t['val']) / np.hypot(0.5 * (a['up'] + a['down']), 0.5 * (t['up'] + t['down']))


def z_split(a, t, n=20_000_000):
    def draw(m):
        u = RNG.standard_normal(n)
        return m['val'] + np.where(u > 0, m['up'], m['down']) * u
    d = draw(a) - draw(t)
    p1 = float(np.mean(d <= 0))
    return float(stats.norm.isf(p1)), p1                        # one-sided p -> z (equals the two-sided z of 2 p1)


def z_barlow(a, t):
    def chi2(mu):
        c = 0.0
        for m in (a, t):
            V = m['up'] * m['down'] + (m['up'] - m['down']) * (mu - m['val'])
            c += (mu - m['val']) ** 2 / max(V, 1e-12)
        return c
    r = minimize_scalar(chi2, bounds=(t['val'], a['val']), method='bounded')
    return float(np.sqrt(r.fun)), float(r.x)


def row(label, a, t, extra=''):
    z_a, z_b = z_paper(a, t), z_sym(a, t)
    z_c, p1 = z_split(a, t)
    z_d, mu = z_barlow(a, t)
    p2 = 2 * stats.norm.sf(z_a)
    print(f'{label:46s} ANNIE {a["val"]:.3f} +{a["up"]:.3f}/-{a["down"]:.3f}  T2K {t["val"]:.3f} +{t["up"]:.3f}/-{t["down"]:.3f} | '
          f'(a) {z_a:.2f}  (b) {z_b:.2f}  (c) {z_c:.2f}  (d) {z_d:.2f} | p(a) = {p2:.2f}' + extra)
    return dict(annie=a, t2k=t, z_paper=z_a, z_sym=z_b, z_split=z_c, p_split_one_sided=p1, z_barlow=z_d, mu_barlow=mu,
                p_two_sided_paper=p2)


out = dict(provenance=provenance(dict(script='10_t2k_comparison.py')), t2k_input=T2K)
print(f'gap {A["val"] - T2K["val"]:.3f}; 1 sigma bars: ANNIE lower edge {A["val"] - A["down"]:.3f}, T2K upper edge '
      f'{T2K["val"] + T2K["up"]:.3f} -> bars {"overlap" if A["val"] - A["down"] < T2K["val"] + T2K["up"] else "do not overlap"}; '
      f'linear sum of the facing bars {A["down"] + T2K["up"]:.3f} vs quadrature {np.hypot(A["down"], T2K["up"]):.3f}\n')
out['nominal'] = row('nominal (paper CV, v3 budget)', A, T2K)

# (e) flux correction: model ratio of flux averages, BNB FV flux vs T2K SK FHC flux (run conditions, to 10 GeV)
g = P['genie_spline']
E_g = np.asarray(g['E_p']); xs_g = np.asarray(g['xs_p']) + np.interp(E_g, g['E_n'], g['xs_n'])
fv_e = np.asarray(P['fv_flux']['edges']); fv = np.asarray(P['fv_flux']['numu']) + np.asarray(P['fv_flux']['nue'])
t_e, t_fl = read_t2k('t2kflux_2020_plus250kA_runcond_sk.txt')[:2]
Es = np.array([s['E'] for s in P['neut_scan']]); sn = np.array([s['sigma'] for s in P['neut_scan']]) / 1e-38


def favg(edges, flux, E, xs, flat_above=None):
    c = 0.5 * (edges[1:] + edges[:-1])
    x = np.interp(c, np.concatenate(([0.0], E)), np.concatenate(([0.0], xs)))     # sigma(0) = 0, constant beyond the table
    return float((x * flux).sum() / flux.sum())


ratio = {}
for name, (E, xs) in (('GENIE spline', (E_g, xs_g)), ('NEUT scan', (Es, sn))):
    s_b, s_t = favg(fv_e, fv, E, xs), favg(t_e, t_fl, E, xs)
    ratio[name] = dict(bnb=s_b, t2k=s_t, ratio=s_b / s_t)
    print(f'flux averages, {name:12s}: BNB FV {s_b:.4f}, T2K SK {s_t:.4f} -> BNB / T2K = {s_b / s_t:.4f}')
frac_above3 = float(t_fl[0.5 * (t_e[1:] + t_e[:-1]) > 3.0].sum() / t_fl.sum())
print(f'  (T2K flux above 3 GeV, where the NEUT scan is held constant: {100 * frac_above3:.1f} %)\n')
k = ratio['GENIE spline']['ratio']
T2K_at_BNB = {kk: v * k for kk, v in T2K.items()}
out['flux_ratio'] = ratio
out['flux_corrected'] = row('(e) T2K moved to the BNB flux (GENIE ratio)', A, T2K_at_BNB)

# (f) NCQE-like signal definition
A_like = {kk: v * (1 + NCQE_LIKE_SHIFT) for kk, v in A.items()}
out['ncqe_like'] = row('(f) ANNIE NCQE-like (NC 2p2h as signal)', A_like, T2K)

# (g) T2K-like de-excitation of the "other" states: CV moves by the 06 shift, the one-sided term leaves the budget
sysd = B['systematics']
up_g, dn_g = quad([v[0] for k_, v in sysd.items() if k_ != 'primary_others'],
                  [v[1] for k_, v in sysd.items() if k_ != 'primary_others'])
val_g = A['val'] * (1 + R06['others'] / 100)
A_g = dict(val=val_g, up=val_g * np.hypot(up_g, B['stat']) / 100, down=val_g * np.hypot(dn_g, B['stat']) / 100)
out['t2k_like_deexcitation'] = row('(g) ANNIE with T2K-like "other" states', A_g, T2K)
A_gf = dict(A_g)
out['t2k_like_deexcitation_flux_corrected'] = row('(g)+(e) both', A_g, T2K_at_BNB)

json.dump(out, open(os.path.join(paths.RESULTS_DIR, '10_t2k_comparison.json'), 'w'), indent=1, default=float)
print('\nwrote', os.path.join(paths.RESULTS_DIR, '10_t2k_comparison.json'))
