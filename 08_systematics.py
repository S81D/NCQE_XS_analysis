#!/usr/bin/env python
"""
08_systematics.py -- inputs of the adopted extraction and the systematic uncertainties evaluated with it (v3).

Run before 06 and 07:   python3 08_systematics.py [--refresh]      (--refresh rebuilds every cache)
Outputs:                results/08_systematics.json (read by 06 and 07), results/08_systematics_log.txt
Runtime (14 workers):   ~7 min without caches (the 44-knob pass dominates), ~10 s with them.

Stages
  A  Inputs that notebook 05 does not provide: N_obs, the pre-beam CIT estimate (03 convention), the OOFV split into
     nuNCQE / other, the true FV nuNCQE count (-> MC-implied <sigma_GENIE>), for the Empirical and base GENIE samples.
  B  Sidebands with the pre-beam CIT: skyshine re-fit (04 logic, CIT replaced by the pre-beam rate), and the joint fit
     of f and alpha_dirt to the signal region and the first-26-bunch >2 sigma sideband (External uncertainty).
  C  Reweighting: interaction model (44-knob GENIE multisim, MicroBooNE 1 sigma), flux (13 knobs), GENIE unisims,
     MC statistics. Budget values use the ratio-transfer method (as quoted in the paper); the event-level evaluation
     on the frozen CV events is printed as a cross-check.
  D  Bunch timing: common (data + MC) and relative (data only) shifts of the CCinc -> nuNCQE map by +-0.5 ns, and the
     bunch width (widening preferred by the in-situ Delta-t fit; narrowing by contraction to the throughgoing-muon
     sigma_BRF).
  E  Background normalisation: CIT (pre-beam) statistics, skyshine, External.
Terms analysed elsewhere (secondary gamma, PMT tilt, energy scale, vertex, POT) are entered in 07.

Every term is the fractional change of sigma = f <sigma_GENIE> obtained by re-extracting f with paper Eq. 26.
Ported (2026-10-15) from the review scripts in NCQE_Paper_Box/Working_Directory_Agent (xs_oofv_prebeam/,
syst_reevaluation/final_44knob/, committee_review_20261015/); every stage prints a closure check against them.
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit, minimize
from concurrent.futures import ProcessPoolExecutor

from lib import paths
from lib import selection as SEL
from lib import extraction as X
from lib.reweight import reweight_file, safe, FLUX_KNOBS, UNISIM_SLOTS
from lib.constants import (ALPHA_SCALING, MC_POT_Empirical, MC_POT_Base, EXPOSURE_SCALE, bunch_sigma,
                           bunch_offset_mapping, centers, centers_data, bunch_time_start, bunch_time_start_data,
                           bunch_time_end_data, MC_third_spill_cut, data_third_spill_cut)

REFRESH = '--refresh' in sys.argv
LOG_LINES = []
RNG = np.random.default_rng(20261015)

# Reference values (frozen CV, Oct 2026) printed next to every recomputed number
REF = dict(N_obs=772, pre_raw=67, sig=152.61, OOFV_nu=140.38, NCother=47.76, CC=7.00, nubar=2.91, OOFV_other=113.01,
           Ext=78.23, T=11324.4, f=1.4603, sigma_genie=2.0513, N_sky=13.95, N_sky_err=12.87, alpha_joint=0.347,
           interaction=8.60, flux=9.95, mc_stat=5.18)

# Superimposed bunch widths [ns] (notebooks 01 / 02; thesis Ch. 4 table): data per epoch FY22, FY23; MC.
# sigma_BRF = sqrt(<data>^2 - <MC>^2) with inverse-variance weighted averages (paper Eq. 21).
BUNCH_WIDTHS = dict(ccinc_data=[(2.83, 0.06), (2.69, 0.09)], thru_data=[(2.38, 0.06), (2.45, 0.09)],
                    ccinc_mc=[(1.99, 0.05)], thru_mc=[(1.92, 0.09)])


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG_LINES.append(s)


def pct(x):
    return 100.0 * float(x)


def updown(*shifts):
    """(up, down) magnitudes in % from a set of signed fractional shifts of sigma."""
    s = np.array(shifts, dtype=float)
    return pct(max(s.max(), 0.0)), pct(abs(min(s.min(), 0.0)))


def summ(d, nboot=5000):
    d = np.asarray(d, float)
    b = np.array([np.sqrt(np.mean(d[RNG.integers(0, len(d), len(d))] ** 2)) for _ in range(nboot)])
    return dict(rms=pct(np.sqrt(np.mean(d ** 2))), rms_err=pct(b.std()), mean=pct(d.mean()), std=pct(d.std()),
                p16=pct(np.percentile(d, 16)), p84=pct(np.percentile(d, 84)))


def fmt(s):
    return (f"RMS about CV {s['rms']:5.2f}% (+-{s['rms_err']:.2f} from 100 universes); mean {s['mean']:+.2f}, "
            f"std {s['std']:.2f}, 16/84%: {s['p16']:+.2f} / {s['p84']:+.2f}")


# =====================================================================================================================
def stage_a():
    log('=' * 110 + '\nA. INPUTS OF THE ADOPTED EXTRACTION')
    d = SEL.data_selection(REFRESH, log)
    mcE = SEL.mc_selection('Empirical', paths.MC_EMPIRICAL_DIR, REFRESH, log)
    mcB = SEL.mc_selection('Base', paths.MC_BASE_DIR, REFRESH, log)
    pot = d['pot']
    fE, fB = pot / (MC_POT_Empirical / 1e20), pot / (MC_POT_Base / 1e20)
    YE, YB = X.yields(mcE, fE), X.yields(mcB, fB)
    d_cit = d['pre_raw'] * SEL.LIVE_SIG / (SEL.PREBEAM[1] - SEL.PREBEAM[0])
    d_cit_err = np.sqrt(d['pre_raw']) * SEL.LIVE_SIG / (SEL.PREBEAM[1] - SEL.PREBEAM[0])
    log(f"  data: {d['n_files']} files, POT {pot:.6f}e20, triggers {d['trig']}; N_obs = {d['n_obs']} (ref {REF['N_obs']}); "
        f"pre-beam [0,150) ns: {d['pre_raw']} (ref {REF['pre_raw']}) x 182/150 = {d_cit:.2f} +/- {d_cit_err:.2f}")
    log('  Empirical MC yields, first 26 bunches, POT-scaled (External x alpha = 0.27); reference = paper Table V:')
    for c, v in YE.items():
        log(f'    {c:11s} {v:9.3f}   (ref {REF[c]:7.2f})')
    TE, TB = mcE['truth'][0] * fE, mcB['truth'][0] * fB
    sgE, sgB = X.sigma_genie(TE, pot), X.sigma_genie(TB, pot)
    log(f'  true FV nuNCQE (POT-scaled): Empirical {TE:.1f} (raw {mcE["truth"][1]:.0f}; ref {REF["T"]}), base {TB:.1f}; '
        f'<sigma_GENIE> = {sgE:.4f} / {sgB:.4f} e-38 (spline average {X.SIGMA_GENIE_SPLINE})')
    return dict(data=d, mcE=mcE, mcB=mcB, pot=pot, fE=fE, fB=fB, YE=YE, YB=YB, TE=TE, TB=TB, sgE=sgE, sgB=sgB,
                d_cit=d_cit, d_cit_err=d_cit_err)


# =====================================================================================================================
def delayed_linear(t, C, m, t0):
    return np.where(t < t0, C, C + m * (t - t0))


def fit_sideband(bins, pred_t, pred_w, data_t):
    """Notebook-04 skyshine fit: piecewise-linear fits to data and prediction; a = excess slope / bin width."""
    cb = 0.5 * (bins[1:] + bins[:-1]); width = bins[1] - bins[0]
    dc, _ = np.histogram(data_t, bins)
    de = np.sqrt(dc).astype(float); de[de == 0] = 1.0
    pc = np.zeros(len(dc)); pe2 = np.zeros(len(dc))
    for a, w in zip(pred_t, pred_w):
        pc += np.histogram(a, bins, weights=w)[0]; pe2 += np.histogram(a, bins, weights=w ** 2)[0]
    pe = np.sqrt(pe2); pe[pe == 0] = 1.0
    pd_, cd = curve_fit(delayed_linear, cb, dc, sigma=de, absolute_sigma=True, p0=[100.0, 0.2, 400.0],
                        bounds=([0, 0, 0], [np.inf, np.inf, bins[-1]]))
    pp, cp = curve_fit(delayed_linear, cb, pc, sigma=pe, absolute_sigma=True, p0=[100.0, 0.0, 400.0],
                       bounds=([0, -np.inf, 0], [np.inf, np.inf, bins[-1]]))
    ok = dc > 0
    chi2 = float((((dc - delayed_linear(cb, *pd_)) / np.sqrt(np.where(ok, dc, 1))) [ok] ** 2).sum())
    dm, dm_err = pd_[1] - pp[1], np.sqrt(cd[1, 1] + cp[1, 1])
    return dict(t0=pd_[2], t0_err=np.sqrt(cd[2, 2]), m_data=pd_[1], m_data_err=np.sqrt(cd[1, 1]), m_pred=pp[1],
                a=dm / width, a_err=dm_err / width, chi2=chi2, ndof=int(ok.sum() - 3))


def n_sky(t_cut, t0, t0_err, a, a_err):
    """N = a (t_cut - t0)^2 / 2 and its error (a and t0 in quadrature)."""
    if t_cut <= t0:
        return 0.0, 0.0
    dd = t_cut - t0
    return 0.5 * a * dd ** 2, np.sqrt((0.5 * dd ** 2 * a_err) ** 2 + (a * dd * t0_err) ** 2)


def stage_b(A):
    log('\n' + '=' * 110 + '\nB. SIDEBANDS WITH THE PRE-BEAM CIT')
    mc, fE, n_pre = A['mcE'], A['fE'], A['data']['pre_raw']
    t_spill = A['data']['t_spill']
    side_d = SEL.abs_dt(t_spill, centers_data) >= 2 * bunch_sigma
    t_data = t_spill[side_d] - bunch_time_start_data
    side_mc = SEL.abs_dt(mc['t'], centers) >= 2 * bunch_sigma
    t_mc = mc['t'] - bunch_time_start
    order = [5, 4, 2, 1, 3, 0]                                     # 04 stack order (External first)
    mc_t = [t_mc[side_mc & (mc['cat'] == c)] for c in order]
    mc_w = [mc['w'][side_mc & (mc['cat'] == c)] * fE * (ALPHA_SCALING if c == 5 else 1.0) for c in order]
    # binning of 04: five bunch periods per bin, first bunch excluded
    t_skip = (centers_data[1] - bunch_offset_mapping - bunch_sigma) - bunch_time_start_data
    bw = 18.936 * 5
    b0 = t_skip + bunch_sigma
    b_end = bunch_time_end_data - bunch_time_start_data
    bins = b0 + np.arange(int(np.floor((b_end - b0) / bw)) + 1) * bw
    # flat pre-beam component: rate n_pre / 150 ns times the sideband livetime of each bin
    step = 0.001
    g = np.arange(bins[0], bins[-1], step)
    gt = g + bunch_time_start_data
    live_ok = ~(gt < bunch_time_start_data) & ~(gt > bunch_time_end_data) & (SEL.abs_dt(gt, centers_data) >= 2 * bunch_sigma)
    live_bin = np.histogram(g[live_ok], bins)[0] * step
    cit_t = np.repeat(0.5 * (bins[1:] + bins[:-1]), n_pre)         # n_pre pseudo-events per bin -> per-bin stat error
    cit_w = np.repeat(live_bin / (SEL.PREBEAM[1] - SEL.PREBEAM[0]), n_pre)
    keep = lambda a: a >= t_skip
    pred_t = [a[keep(a)] for a in mc_t] + [cit_t[keep(cit_t)]]
    pred_w = [w[keep(a)] for a, w in zip(mc_t, mc_w)] + [cit_w[keep(cit_t)]]
    fit = fit_sideband(bins, pred_t, pred_w, t_data[keep(t_data)])
    t_cut = data_third_spill_cut - bunch_time_start_data
    nsb, nsb_err = n_sky(t_cut, fit['t0'], fit['t0_err'], fit['a'], fit['a_err'])
    N_sky, N_sky_err = nsb * EXPOSURE_SCALE, nsb_err * EXPOSURE_SCALE
    log(f"  skyshine fit ({len(bins) - 1} bins of {bw:.2f} ns): t0 = {fit['t0']:.1f} +/- {fit['t0_err']:.1f} ns, "
        f"m_data = {fit['m_data']:.4f} +/- {fit['m_data_err']:.4f}, m_pred = {fit['m_pred']:.4f}, "
        f"a = {fit['a'] * 1e3:.3f} +/- {fit['a_err'] * 1e3:.3f} e-3 ns^-2, chi2/dof {fit['chi2']:.1f}/{fit['ndof']}")
    log(f"  skyshine in the signal region: sideband {nsb:.2f} +/- {nsb_err:.2f} x {EXPOSURE_SCALE:.4f} = "
        f"{N_sky:.2f} +/- {N_sky_err:.2f} (ref {REF['N_sky']} +/- {REF['N_sky_err']})")
    # ---- joint fit of f and alpha: signal region + first-26-bunch >2 sigma sideband (disjoint data samples)
    in26_d = t_data <= t_cut
    in26_mc = side_mc & ~(mc['t'] > MC_third_spill_cut)
    y = {c: float(mc['w'][in26_mc & (mc['cat'] == c)].sum() * fE) for c in range(6)}
    y_onu = float(mc['w'][in26_mc & mc['oofv_nu']].sum() * fE)
    g26 = np.arange(bunch_time_start_data, data_third_spill_cut, step)
    live26 = float((SEL.abs_dt(g26, centers_data) >= 2 * bunch_sigma).sum() * step)
    pre26 = n_pre * live26 / (SEL.PREBEAM[1] - SEL.PREBEAM[0])
    Y = A['YE']
    ext_sig = Y['Ext'] / ALPHA_SCALING
    rest_sig = Y['NCother'] + Y['CC'] + Y['nubar'] + Y['OOFV_other'] + A['d_cit'] + N_sky
    nu_sb = y[0] + y_onu
    rest_sb = sum(y[c] for c in range(5)) - nu_sb + pre26 + nsb
    n_sb = int(in26_d.sum())
    Am = np.array([[Y['sig'] + Y['OOFV_nu'], ext_sig], [nu_sb, y[5]]])
    rhs = np.array([A['data']['n_obs'] - rest_sig, n_sb - rest_sb])
    f_j, a_j = np.linalg.solve(Am, rhs)
    Ai = np.linalg.inv(Am)
    cov = Ai @ np.diag([A['data']['n_obs'], n_sb]) @ Ai.T
    log(f"  joint fit:  signal   {A['data']['n_obs']} = f {Y['sig'] + Y['OOFV_nu']:.2f} + alpha {ext_sig:.2f} + {rest_sig:.2f}")
    log(f"              sideband {n_sb} = f {nu_sb:.2f} + alpha {y[5]:.2f} + {rest_sb:.2f}   (CIT {pre26:.2f}, skyshine {nsb:.2f})")
    log(f"  -> f = {f_j:.3f} +/- {np.sqrt(cov[0, 0]):.3f}, alpha = {a_j:.3f} +/- {np.sqrt(cov[1, 1]):.3f} (data stat.), "
        f"correlation {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):+.2f}  (ref alpha {REF['alpha_joint']})")
    return dict(N_sky=N_sky, N_sky_err=N_sky_err, N_sky_sideband=nsb, fit={k: float(v) for k, v in fit.items()},
                joint=dict(f=float(f_j), f_err=float(np.sqrt(cov[0, 0])), alpha=float(a_j), alpha_err=float(np.sqrt(cov[1, 1])),
                           corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))))


# =====================================================================================================================
def run_reweight_pass(mcE):
    """One pass over the 44-knob production (reselection + weights + truth sums), cached in cache/reweight_44knob.npz."""
    cache = os.path.join(paths.CACHE_DIR, 'reweight_44knob.npz')
    names44 = sorted(f for f in os.listdir(paths.MC_44KNOB_DIR) if f.endswith('.root'))
    if os.path.exists(cache) and not REFRESH:
        z = dict(np.load(cache, allow_pickle=False))
        if list(z['names44']) == names44:
            log(f'  44-knob production: from cache ({len(names44)} files)')
            return z
    nom_names = list(mcE['files'])
    iw = mcE['in_win']
    ce_by_name = {nom_names[i]: mcE['ce'][iw & (mcE['fidx'] == i)] for i in np.unique(mcE['fidx'][iw])}
    jobs = [(n, ce_by_name.get(n, np.zeros(0, dtype=np.int64)) if n in nom_names else None) for n in names44]
    log(f'  44-knob production: {len(names44)} files in {paths.MC_44KNOB_DIR} ...')
    with ProcessPoolExecutor(max_workers=paths.N_WORKERS) as ex:
        res = list(ex.map(reweight_file, jobs, chunksize=8))
    Z = dict(names44=np.array(names44), ok=np.array([r['ok'] for r in res]), matched=np.array([r['matched'] for r in res]),
             why=np.array([r.get('why', '') for r in res]))
    good = [r for r in res if r['ok']]
    for k in ('cv', 'cv2', 'n'):
        Z[f'T_{k}'] = np.sum([r['T'][k] for r in good])
    for k in ('x', 'f', 'k', 'u'):
        Z[f'T_{k}'] = np.sum([r['T'][k] for r in good], axis=0)
    rs = [r['sel'] for r in good if 'sel' in r]
    for k in rs[0]:
        Z[f'R_{k}'] = np.concatenate([s[k] for s in rs], axis=1 if k in ('kn', 'un') else 0)
    # event level: nominal in-window events of matched files, in the order of mcE (file index, then entry)
    nom_w = {r['name']: r['nom'] for r in res if r.get('matched')}
    idx_all = np.where(iw)[0]
    match = np.zeros(len(idx_all), bool)
    a0 = np.full((len(idx_all), 100), np.nan); kn = np.full((len(FLUX_KNOBS), len(idx_all), 100), np.nan)
    un = np.full((len(UNISIM_SLOTS), len(idx_all), 2), np.nan); cvr = np.full(len(idx_all), np.nan)
    for name, wv in nom_w.items():
        fi = nom_names.index(name)
        rows = np.where(mcE['fidx'][idx_all] == fi)[0]
        a0[rows], kn[:, rows], un[:, rows], cvr[rows], match[rows] = wv['a0'], wv['kn'], wv['un'], wv['cvraw'], True
    Z.update(E_idx=idx_all, E_match=match, E_a0=a0, E_kn=kn, E_un=un, E_cvraw=cvr)
    np.savez(cache, **Z)
    return Z


def stage_c(A, B):
    log('\n' + '=' * 110 + '\nC. REWEIGHTING: INTERACTION MODEL, FLUX, UNISIMS, MC STATISTICS')
    mcE, fE, N, D, Dsky = A['mcE'], A['fE'], A['data']['n_obs'], A['d_cit'], B['N_sky']
    Y0 = A['YE']
    f_cv = X.extract_f(Y0, N, D, Dsky)
    Z = run_reweight_pass(mcE)
    nok, nmatch = int(Z['ok'].sum()), int(Z['matched'].sum())
    bad = [f'{n} ({w})' for n, w, o in zip(Z['names44'], Z['why'], Z['ok']) if not o]
    log(f'  production files read: {nok}/{len(Z["names44"])} (skipped: {", ".join(bad) or "none"}); '
        f'matched event-by-event to the CV sample: {nmatch}')
    # ---------------- ratio transfer (budget): universe/CV ratio per category on the reselected production
    R = {k[2:]: Z[k] for k in Z if k.startswith('R_')}
    Rsel = dict(cat=R['cat'], oofv_nu=R['oofv_nu'], w=R['w'], in_win=R['in_win'], in26=R['in26'])
    cvr = np.where(np.isfinite(R['cvraw']), R['cvraw'], np.nan)
    W = dict(x=safe(R['a0']), f=safe(np.prod(R['kn'].astype(np.float64), axis=0) * cvr[:, None]))
    W['k'] = [safe(R['kn'][i].astype(np.float64) * cvr[:, None]) for i in range(len(FLUX_KNOBS))]
    W['u'] = [safe(R['un'][j]) for j in range(len(UNISIM_SLOTS))]
    Tcv = float(Z['T_cv'])
    yc_R = X.yields(Rsel, 1.0)

    def rt_shift(Wu, Tu, genie, cats=None, cfg='adopted'):
        yu = X.yields(Rsel, 1.0, w=Wu)
        U = Wu.shape[1]
        Yu = {c: Y0[c] * (yu[c] / yc_R[c] if (cats is None or c in cats) else np.ones(U)) for c in Y0}
        Yu['Ext'] = Y0['Ext'] * np.ones(U)                         # External held at alpha x CV (data-constrained)
        g = (Tu / Tcv) if (genie and (cats is None or 'sig' in cats)) else 1.0
        if cfg == 'adopted':
            return X.extract_f(Yu, N, D, Dsky) * g / f_cv - 1
        return X.extract_f_earlier(Yu, N) * g / X.extract_f_earlier(Y0, N) - 1

    out = dict(method='ratio transfer')
    dx, dfl = rt_shift(W['x'], Z['T_x'], True), rt_shift(W['f'], Z['T_f'], False)
    out['interaction'], out['flux'] = summ(dx), summ(dfl)
    log(f"  interaction model (44 knobs): {fmt(out['interaction'])}   (ref {REF['interaction']})")
    log(f"  flux (13 knobs):              {fmt(out['flux'])}   (ref {REF['flux']})")
    log(f'    mean T_u/T_CV: GENIE {np.mean(Z["T_x"] / Tcv):.4f}, flux {np.mean(Z["T_f"] / Tcv):.4f}')
    parts = {}
    for nm, cs in (('signal-like (FV + OOFV nuNCQE)', ['sig', 'OOFV_nu']), ('NCother', ['NCother']),
                   ('OOFV other', ['OOFV_other']), ('CC', ['CC']), ('nubarNCQE', ['nubar']),
                   ('all MC backgrounds', ['NCother', 'CC', 'nubar', 'OOFV_other'])):
        parts[nm] = summ(rt_shift(W['x'], Z['T_x'], True, cats=cs))['rms']
        log(f'    interaction, only {nm:32s} varied: {parts[nm]:5.2f}%')
    out['interaction_parts'] = parts
    out['flux_knobs'] = {k.split('_')[0]: summ(rt_shift(W['k'][i], Z['T_k'][i], False))['rms'] for i, k in enumerate(FLUX_KNOBS)}
    log('    flux by knob: ' + ', '.join(f'{k} {v:.2f}' for k, v in out['flux_knobs'].items()))
    lab = {'NormNCMEC': '+100%/-100%', 'MaNCEL': '+1s/-1s', 'EtaNCEL': '+1s/-1s', 'MaCCQE': '+1s/-1s',
           'CoulombCCQE': '+1s/-1s', 'RPA_CCQE': '+1s/-1s', 'AxFFCCQEshape': '1/0', 'VecFFCCQEshape': '1/0'}
    out['unisims'] = {}
    for j, dname in enumerate(UNISIM_SLOTS):
        s = rt_shift(W['u'][j], Z['T_u'][j], True)
        rate = Z['T_u'][j] / Tcv - 1
        out['unisims'][dname] = [pct(s[0]), pct(s[1])]
        log(f'    unisim {dname:15s} [{lab[dname]:>11s}] {pct(s[0]):+6.2f} / {pct(s[1]):+6.2f} %   '
            f'(true FV nuNCQE rate {pct(rate[0]):+5.1f} / {pct(rate[1]):+5.1f} %)')
    ex = dict(interaction=summ(rt_shift(W['x'], Z['T_x'], True, cfg='earlier'))['rms'],
              flux=summ(rt_shift(W['f'], Z['T_f'], False, cfg='earlier'))['rms'])
    out['earlier_configuration'] = ex
    log(f"  same weights, earlier configuration (OOFV nuNCQE as background, off-beam CIT): interaction {ex['interaction']:.2f}%, "
        f"flux {ex['flux']:.2f}%  -> the configuration, not the arithmetic, inflated the old 24.8 / 24.6%")
    # ---------------- event level on the frozen CV events (cross-check)
    iw = mcE['in_win']
    E_rows = Z['E_idx']
    m = Z['E_match']

    def ev_weights(arr):                                          # (n_inwin, U): universe weight if matched, else CV
        U = arr.shape[1]
        Wf = np.repeat(mcE['w'][E_rows][:, None], U, axis=1)
        Wf[m] = arr[m]
        Wall = np.repeat(mcE['w'][:, None], U, axis=1)
        Wall[E_rows] = Wf
        return Wall
    ecv = np.where(np.isfinite(Z['E_cvraw']), Z['E_cvraw'], np.nan)
    ex_x = ev_weights(safe(Z['E_a0']))
    ex_f = ev_weights(safe(np.prod(Z['E_kn'].astype(np.float64), axis=0) * ecv[:, None]))

    def ev_shift(Wall, Tu, genie):
        Yu = X.yields(mcE, fE, w=Wall)
        Yu['Ext'] = Y0['Ext'] * np.ones(Wall.shape[1])
        return X.extract_f(Yu, N, D, Dsky) * ((Tu / Tcv) if genie else 1.0) / f_cv - 1
    out['event_level'] = dict(interaction=summ(ev_shift(ex_x, Z['T_x'], True)), flux=summ(ev_shift(ex_f, Z['T_f'], False)),
                              cv_events_matched=f'{int((m & mcE["in26"][E_rows]).sum())}/{int(mcE["in26"][E_rows].sum())}')
    log(f"  cross-check, event level on the frozen CV events ({out['event_level']['cv_events_matched']} first-26-bunch events "
        f"matched): interaction {out['event_level']['interaction']['rms']:.2f}%, flux {out['event_level']['flux']['rms']:.2f}%")
    # ---------------- MC statistics (delta method on the CV sample; Poisson bootstrap cross-check)
    cv = SEL.cv_mask(mcE)
    w = np.where(np.isfinite(mcE['w']), mcE['w'], 0.0)
    cat, onu = mcE['cat'], mcE['oofv_nu']
    var = lambda mm, sc=1.0: (sc * fE) ** 2 * (w[cv & mm] ** 2).sum()
    vB = sum(var(mm) for mm in (cat == 1, cat == 2, cat == 3, (cat == 4) & ~onu)) + var(cat == 5, ALPHA_SCALING)
    M = Y0['sig'] + Y0['OOFV_nu']
    nb = N - D - Dsky - X.m_bkg(Y0)
    T = A['TE']
    vU = fE ** 2 * mcE['truth'][5] - var(cat == 0)
    terms = dict(backgrounds=np.sqrt(vB) / nb, FV_signal=abs(1 / T - 1 / M) * np.sqrt(var(cat == 0)),
                 OOFV_nuNCQE=np.sqrt(var(onu)) / M, truth_denominator=np.sqrt(vU) / T)
    rel = np.sqrt(sum(v ** 2 for v in terms.values()))
    sel_i = np.where(cv)[0]
    boots = []
    for _ in range(4000):
        p = np.zeros(len(w)); p[sel_i] = RNG.poisson(1.0, len(sel_i))
        Yb = X.yields(mcE, fE, w=w * p)
        Tb = T - fE * w[cv & (cat == 0)].sum() + RNG.normal(0, np.sqrt(max(vU, 0))) + Yb['sig']
        boots.append(X.extract_f(Yb, N, D, Dsky) * Tb / (f_cv * T) - 1)
    out['mc_stat'] = dict(delta=pct(rel), terms={k: pct(v) for k, v in terms.items()}, bootstrap_std=pct(np.std(boots)))
    log(f"  MC statistics: {pct(rel):.2f}% (delta method; ref {REF['mc_stat']}), bootstrap {pct(np.std(boots)):.2f}%; "
        + ', '.join(f'{k} {pct(v):.2f}' for k, v in terms.items()))
    return out


# =====================================================================================================================
def stage_d(A, B):
    log('\n' + '=' * 110 + '\nD. BUNCH TIMING')
    mc, fE, D, Dsky = A['mcE'], A['fE'], A['d_cit'], B['N_sky']
    D_err, Dsky_err = A['d_cit_err'], B['N_sky_err']
    t_spill = A['data']['t_spill']
    HALF = 18.936 / 2
    EDGES = np.arange(-9.0, 9.0001, 0.5)
    CTR = 0.5 * (EDGES[1:] + EDGES[:-1])
    keep = ~(mc['t'] > MC_third_spill_cut) & ~(mc['t'] < bunch_time_start)
    dt_mc = SEL.signed_dt(mc['t'][keep], centers)
    w_mc = (np.where(np.isfinite(mc['w']), mc['w'], 0.0) * fE * np.where(mc['cat'] == 5, ALPHA_SCALING, 1.0))[keep]
    ncq = ((mc['cat'] == 0) | mc['oofv_nu'])[keep]
    ext = (mc['cat'] == 5)[keep]
    noise = np.random.default_rng(1).standard_normal((50, len(dt_mc)))

    def mc_hists(d2=0.0, shift=0.0):
        """Simulated bunch-correlated Delta-t: widened by d2 > 0 (convolution, 50 copies per event) or narrowed by
        d2 < 0 (contraction about the bunch centre); moved by -shift (== MC window moved by +shift)."""
        if d2 > 0:
            x = dt_mc[None, :] + np.sqrt(d2) * noise
        else:
            x = (dt_mc * np.sqrt(1 + d2 / bunch_sigma ** 2))[None, :]
        x = (x - shift + HALF) % (2 * HALF) - HALF
        ib = np.digitize(x, EDGES) - 1; ok = (ib >= 0) & (ib < len(CTR))
        frac = np.zeros((len(dt_mc), len(CTR)))
        np.add.at(frac, (np.broadcast_to(np.arange(len(dt_mc)), ib.shape)[ok], ib[ok]), 1.0 / x.shape[0])
        H = {k: (frac * (w_mc * mm)[:, None]).sum(0) for k, mm in (('n', ncq), ('e', ext), ('o', ~ncq & ~ext))}
        E2 = {k: ((frac * w_mc[:, None]) ** 2 * mm[:, None]).sum(0) for k, mm in (('n', ncq), ('e', ext), ('o', ~ncq & ~ext))}
        ins = (np.abs(x) < bunch_sigma).mean(0)
        return H, E2, (w_mc * ins)[ncq].sum(), (w_mc * ins)[~ncq].sum()

    def n_data(delta=0.0):
        return int(SEL.signal_window_data(t_spill, bunch_offset_mapping + delta).sum())

    fw = lambda n, S, O: (n - O - D - Dsky) / S
    _, _, S0, O0 = mc_hists()
    n0 = n_data()
    f0 = fw(n0, S0, O0)
    log(f'  closure: N_obs {n0}, MC signal-like {S0:.2f}, other MC {O0:.2f}, f {f0:.4f}')
    # (1) common and (2) relative shifts of the CCinc -> nuNCQE map by +-0.5 ns
    sh = {}
    for dd in (-0.5, +0.5):
        nd = n_data(dd)
        _, _, Sm, Om = mc_hists(0.0, shift=dd)
        sh[dd] = dict(N=nd, relative=fw(nd, S0, O0) / f0 - 1, common=fw(nd, Sm, Om) / f0 - 1, mc_only=fw(n0, Sm, Om) / f0 - 1)
        log(f"  window shift {dd:+.1f} ns: N_data {nd}; relative (data only) {pct(sh[dd]['relative']):+.2f}%, "
            f"common (data + MC) {pct(sh[dd]['common']):+.2f}%, MC only {pct(sh[dd]['mc_only']):+.2f}%")
    rel = updown(sh[-0.5]['relative'], sh[0.5]['relative'])
    com = updown(sh[-0.5]['common'], sh[0.5]['common'])
    # (3) width: in-situ fit of the Delta-t distribution of the selected events over the full bunch period
    data = t_spill[(t_spill >= centers_data[0] - bunch_offset_mapping - bunch_sigma) & ~(t_spill > data_third_spill_cut)]
    hd = np.histogram(SEL.signed_dt(data, centers_data), EDGES)[0]
    flat = (D + Dsky) / (2 * bunch_sigma) * np.diff(EDGES) * np.where(np.abs(CTR) < bunch_sigma, 1.0, 25 / 26)
    k_flat = np.hypot(D_err, Dsky_err) / (D + Dsky)
    k_ext = abs(B['joint']['alpha'] - ALPHA_SCALING) / ALPHA_SCALING

    def fit(d2, delta, float_ext):
        H, E2, _, _ = mc_hists(d2, shift=delta)

        def chi2(p):
            f, k, ke = p if float_ext else (p[0], p[1], 1.0)
            pr = H['o'] + ke * H['e'] + f * H['n'] + k * flat
            v = pr + E2['o'] + ke ** 2 * E2['e'] + f ** 2 * E2['n']
            c = (((hd - pr) ** 2) / np.maximum(v, 1e-6)).sum() + ((k - 1) / k_flat) ** 2
            return c + (((ke - 1) / k_ext) ** 2 if float_ext else 0.0)
        r = minimize(chi2, [1.5, 1.0, 1.0] if float_ext else [1.5, 1.0], method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        return r.fun
    D2 = np.round(np.arange(-2.0, 6.01, 0.25), 2)
    DEL = np.round(np.arange(-1.0, 0.61, 0.1), 2)
    fwin = np.array([fw(n0, *mc_hists(d2)[2:]) / f0 - 1 for d2 in D2])
    scans = {}
    for float_ext in (False, True):
        c2 = np.array([min(fit(d2, de, float_ext) for de in DEL) for d2 in D2])
        j = int(c2.argmin())
        scans['ext_floated' if float_ext else 'nominal'] = dict(best_d2=float(D2[j]), dchi2_0=float(c2[D2 == 0][0] - c2[j]),
                                                                fwin=float(fwin[j]), chi2=c2.tolist())
        log(f"  width fit ({'External floated by +-' + format(100 * k_ext, '.1f') + '%' if float_ext else 'nominal'}; "
            f"CIT + skyshine constrained to +-{100 * k_flat:.1f}%): best extra variance {D2[j]:+.2f} ns^2 "
            f"({np.sqrt(max(D2[j], 0)):.2f} ns), chi2(0) - min = {c2[D2 == 0][0] - c2[j]:.2f} -> f_window {pct(fwin[j]):+.2f}%")
    wa = lambda v: sum(a / e ** 2 for a, e in v) / sum(e ** -2 for _, e in v)
    brf = lambda dat, mcv: np.sqrt(wa(dat) ** 2 - wa(mcv) ** 2)
    s_nom = brf(BUNCH_WIDTHS['ccinc_data'] + BUNCH_WIDTHS['thru_data'], BUNCH_WIDTHS['ccinc_mc'] + BUNCH_WIDTHS['thru_mc'])
    s_thru = brf(BUNCH_WIDTHS['thru_data'], BUNCH_WIDTHS['thru_mc'])
    d2n = s_thru ** 2 - s_nom ** 2
    narrow = float(np.interp(d2n, D2, fwin))
    mirror = -float(np.interp(-d2n, D2, fwin))
    log(f'  narrowing: sigma_BRF {s_nom:.3f} -> {s_thru:.3f} ns (d2 = {d2n:+.3f} ns^2): contraction {pct(narrow):+.2f}% '
        f'(budget), mirror of the equal widening {pct(mirror):+.2f}% (paper draft 1)')
    width = (pct(scans['nominal']['fwin']), pct(abs(narrow)))
    up = np.sqrt(rel[0] ** 2 + com[0] ** 2 + width[0] ** 2)
    dn = np.sqrt(rel[1] ** 2 + com[1] ** 2 + width[1] ** 2)
    log(f'  bunch timing: common +{com[0]:.2f}/-{com[1]:.2f}, relative +{rel[0]:.2f}/-{rel[1]:.2f}, '
        f'width +{width[0]:.2f}/-{width[1]:.2f}  ->  +{up:.2f} / -{dn:.2f} %')
    return dict(shifts={str(k): {kk: (pct(vv) if kk != 'N' else vv) for kk, vv in v.items()} for k, v in sh.items()},
                common=com, relative=rel, width=width, narrowing_mirror=pct(mirror), sigma_brf=[float(s_nom), float(s_thru)],
                width_scans=scans, d2_grid=D2.tolist(), f_window=fwin.tolist(), total=(float(up), float(dn)))


# =====================================================================================================================
def stage_e(A, B):
    log('\n' + '=' * 110 + '\nE. BACKGROUND NORMALISATION')
    Y0, N, D, Dsky = A['YE'], A['data']['n_obs'], A['d_cit'], B['N_sky']
    f0 = X.extract_f(Y0, N, D, Dsky)
    sh = lambda **kw: X.extract_f(kw.get('Y', Y0), N, kw.get('D', D), kw.get('sky', Dsky)) / f0 - 1
    cit = updown(sh(D=D + A['d_cit_err']), sh(D=D - A['d_cit_err']))
    sky = updown(sh(sky=Dsky + B['N_sky_err']), sh(sky=Dsky - B['N_sky_err']))
    da = abs(B['joint']['alpha'] - ALPHA_SCALING)
    Ya = lambda a: {**Y0, 'Ext': Y0['Ext'] / ALPHA_SCALING * a}
    ext = updown(sh(Y=Ya(ALPHA_SCALING + da)), sh(Y=Ya(ALPHA_SCALING - da)))
    log(f'  CIT (pre-beam) statistics: {D:.2f} +/- {A["d_cit_err"]:.2f} events -> +{cit[0]:.2f} / -{cit[1]:.2f} %')
    log(f'  skyshine: {Dsky:.2f} +/- {B["N_sky_err"]:.2f} events -> +{sky[0]:.2f} / -{sky[1]:.2f} %')
    log(f'  External: alpha {ALPHA_SCALING} +/- {da:.3f} (|alpha_joint - alpha_CCinc|) -> +{ext[0]:.2f} / -{ext[1]:.2f} %')
    return dict(cit_stat=cit, skyshine=sky, external=ext, delta_alpha=float(da))


# =====================================================================================================================
def main():
    t0 = time.time()
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    A = stage_a()
    B = stage_b(A)
    Y0 = A['YE']
    f_cv = X.extract_f(Y0, A['data']['n_obs'], A['d_cit'], B['N_sky'])
    f_base = X.extract_f(A['YB'], A['data']['n_obs'], A['d_cit'], B['N_sky'])
    stat = X.stat_f(Y0, A['data']['n_obs']) / f_cv
    log(f"\n  CV: f = {f_cv:.4f} (ref {REF['f']}), sigma = {f_cv * A['sgE']:.4f} +/- {stat * f_cv * A['sgE']:.4f} (stat, "
        f"{pct(stat):.2f}%) e-38; base GENIE de-excitation: f = {f_base:.4f}, sigma = {f_base * A['sgB']:.4f} e-38 "
        f"({pct(f_base * A['sgB'] / (f_cv * A['sgE']) - 1):+.1f}%)")
    C = stage_c(A, B)
    Dd = stage_d(A, B)
    E = stage_e(A, B)
    budget = {'Interaction model': [C['interaction']['rms']] * 2, 'Flux': [C['flux']['rms']] * 2,
              'MC statistics': [C['mc_stat']['delta']] * 2, 'Bunch timing': list(Dd['total']),
              'External normalization': list(E['external']), 'Skyshine': list(E['skyshine']),
              'Pre-beam (CIT) statistics': list(E['cit_stat'])}
    log('\n' + '=' * 110 + '\nTERMS FOR 07 (% on sigma, up / down)')
    for k, (u, dn) in budget.items():
        log(f'  {k:28s} +{u:5.2f} / -{dn:5.2f}')
    res = dict(
        provenance=X.provenance(dict(script='08_systematics.py', mc_empirical=paths.MC_EMPIRICAL_DIR, mc_base=paths.MC_BASE_DIR,
                                     mc_44knob=paths.MC_44KNOB_DIR, data=paths.DATA_DIR, runtime_s=round(time.time() - t0))),
        inputs=dict(N_obs=A['data']['n_obs'], pre_raw=A['data']['pre_raw'], D_CIT=A['d_cit'], D_CIT_err=A['d_cit_err'],
                    D_sky=B['N_sky'], D_sky_err=B['N_sky_err'], POT_1e20=A['pot'], triggers=A['data']['trig'],
                    alpha_dirt=ALPHA_SCALING, yields={k: float(v) for k, v in Y0.items()},
                    yields_base={k: float(v) for k, v in A['YB'].items()}, N_true_FV=float(A['TE']),
                    N_true_FV_base=float(A['TB']), sigma_genie=float(A['sgE']), sigma_genie_base=float(A['sgB']),
                    phi_fv=X.PHI_FV, n_target=X.N_TARGET, f=float(f_cv), f_base=float(f_base), stat_frac=float(stat)),
        sidebands=B, reweighting=C, bunch_timing=Dd, background=E, budget_terms=budget)
    clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else
                       [clean(v) for v in o] if isinstance(o, (list, tuple)) else
                       o.tolist() if isinstance(o, np.ndarray) else
                       float(o) if isinstance(o, (np.floating, np.integer)) else o)
    with open(os.path.join(paths.RESULTS_DIR, '08_systematics.json'), 'w') as fh:
        json.dump(clean(res), fh, indent=1)
    log(f'\nwrote {paths.RESULTS_DIR}08_systematics.json [{time.time() - t0:.0f} s]')
    with open(os.path.join(paths.RESULTS_DIR, '08_systematics_log.txt'), 'w') as fh:
        fh.write('\n'.join(LOG_LINES) + '\n')


if __name__ == '__main__':
    main()
