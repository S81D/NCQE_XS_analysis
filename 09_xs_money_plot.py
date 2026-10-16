#!/usr/bin/env python
"""
09_xs_money_plot.py -- final nuNCQE cross-section figure (paper Fig. "xs_result"), v3.

Run after 07:   python3 09_xs_money_plot.py [--out plots/NAME.png] [--budget results/07_budget.json]
                                            [--placement median|mean] [--layout column|wide] [--rebuild-inputs]
Defaults reproduce the paper's figure style: median placement, PRD single-column layout.

Two steps, so that the figure is reproducible from this repository alone:
  1. --rebuild-inputs (only needed if the generator outputs change): reduce the generator outputs in
     paths.GENERATORS_DIR (GENIE spline file xs_10GeV.root, FV flux ANNIE_FLUX_r_0.7m_y_1m.root, NEUT 6.1.4 flat trees
     nc_numu_flat.root / nc_nue_flat.root with the generation flux gsimple_hist.true.root, and the NEUT nu_mu energy
     scan scan_output/numu/E*MeV/flat.root; ~43 GB in total) to inputs/xs_predictions.json (~100 kB, tracked), with the
     md5 checksum of every source file.
  2. Draw the figure from inputs/xs_predictions.json, inputs/t2kflux_2020/ and the uncertainty budget (07), and write
     a <name>_values.json sidecar with every plotted number and its provenance.

Physics content (unchanged from the figure of PRD draft 1, NCQE_Paper_Box/Working_Directory_Agent/xs_plot/
xs_money_plot.py, itself from BuildEventGenerators/NEUT vs GENIE NCQE comparison.ipynb):
  ANNIE point   : sigma, stat. and total uncertainties from the budget; base GENIE de-excitation result (open diamond)
  GENIE         : spline nu_mu 16O NC elastic (qel_nc_p + qel_nc_n) vs E; flux-averaged point = MC-implied value
  NEUT 6.1.4    : flux-averaged with the generation flux (as the notebook); energy dependence from the nu_mu scan
  T2K           : PRD 100, 112009: 1.70 +- 0.17 (stat.) +0.51/-0.38 (syst.), total +0.54/-0.42
  placement     : median of each experiment's nu flux, bar = 16th-84th percentiles (lib/flux_energy.py)
"""
import os, sys, json, glob, hashlib, argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

from lib import paths
from lib.flux_energy import summary, read_t2k
from lib.extraction import provenance

PRED = os.path.join(paths.INPUTS_DIR, 'xs_predictions.json')
N_O16_NUCLEONS = 16
# T2K (PRD 100, 112009; checked 2026-10-09 against the text and the vector paths of its Fig. 12):
#   1.70 +/- 0.17 (stat) +0.51/-0.38 (syst) -> total +0.54/-0.42; mean flux energy 0.82 GeV, bar 0.50-1.75 GeV
T2K = dict(val=1.70, err_down=0.42, err_up=0.54, stat=0.17, x=0.82, x_lo=0.50, x_hi=1.75)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================================== step 1
def rebuild_inputs():
    G = paths.GENERATORS_DIR
    src = {}
    with uproot.open(G + 'xs_10GeV.root') as root:
        E_p, xs_p = root['nu_mu_O16/qel_nc_p'].values()
        E_n, xs_n = root['nu_mu_O16/qel_nc_n'].values()
    with uproot.open(G + 'ANNIE_FLUX_r_0.7m_y_1m.root') as r:
        fl_mu, edges = r['numu_cv'].to_numpy()
        fl_e, edges_e = r['nue_cv'].to_numpy()
    assert np.array_equal(edges, edges_e)
    with uproot.open(G + 'gsimple_hist.true.root') as fx:
        gs = {fl: fx[h].to_numpy() for fl, h in (('numu', 'hEnumu_cv'), ('nue', 'hEnue_cv'))}
    fv = {'numu': (fl_mu, edges), 'nue': (fl_e, edges)}
    neut = {}
    for fl, fn in (('numu', 'nc_numu_flat.root'), ('nue', 'nc_nue_flat.root')):
        with uproot.open(G + fn) as f:
            a = f['FlatTree_VARS'].arrays(['flagNCEL', 'fScaleFactor', 'Weight', 'Enu_true'], library='np')
        assert np.all(a['Weight'] == 1.0)
        sf = np.unique(a['fScaleFactor'])[0]
        sel = a['flagNCEL'].astype(bool)
        sigma_gs = N_O16_NUCLEONS * sel.sum() * sf                   # generation (gsimple) flux, notebook convention
        gv, ge = gs[fl]; fvv, fe = fv[fl]                             # reweight to the FV flux (densities per GeV)
        d_gs = gv / np.diff(ge) / gv.sum(); d_fv = fvv / np.diff(fe) / fvv.sum()
        E = a['Enu_true'][sel]
        num = d_fv[np.clip(np.digitize(E, fe) - 1, 0, len(d_fv) - 1)]
        den = d_gs[np.clip(np.digitize(E, ge) - 1, 0, len(d_gs) - 1)]
        w = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        neut[fl] = dict(gs=float(sigma_gs), fv=float(N_O16_NUCLEONS * sf * w.sum()), phi_gs=float(gv.sum()),
                        phi_fv=float(fvv.sum()), n_ncel=int(sel.sum()), scale_factor=float(sf))
    scan = []
    for dd in sorted(glob.glob(G + 'scan_output/numu/E*MeV')):
        p = os.path.join(dd, 'flat.root')
        with uproot.open(p) as f:
            if 'FlatTree_VARS' not in f.keys(cycle=False):
                continue
            a = f['FlatTree_VARS'].arrays(['flagNCEL', 'fScaleFactor', 'Weight'], library='np')
        scan.append(dict(E=int(os.path.basename(dd)[1:-3]) / 1000.0,
                         sigma=float(N_O16_NUCLEONS * int(a['flagNCEL'].sum()) * np.unique(a['fScaleFactor'])[0]),
                         file=os.path.relpath(p, G), md5=md5(p)))
        print(f'  NEUT scan {scan[-1]["E"]:.2f} GeV: {scan[-1]["sigma"] / 1e-38:.4f} e-38', flush=True)
    for name in ('xs_10GeV.root', 'ANNIE_FLUX_r_0.7m_y_1m.root', 'gsimple_hist.true.root', 'nc_numu_flat.root',
                 'nc_nue_flat.root'):
        src[name] = dict(md5=md5(G + name), bytes=os.path.getsize(G + name))
    out = dict(provenance=provenance(dict(script='09_xs_money_plot.py --rebuild-inputs', generators_dir=G, sources=src)),
               genie_spline=dict(E_p=list(map(float, E_p)), xs_p=list(map(float, xs_p)), E_n=list(map(float, E_n)),
                                 xs_n=list(map(float, xs_n)), note='nu_mu_O16 qel_nc_p / qel_nc_n, 1e-38 cm^2 per nucleus'),
               fv_flux=dict(edges=edges.tolist(), numu=fl_mu.tolist(), nue=fl_e.tolist(),
                            note='ANNIE FV (r < 0.7 m, |y| < 1 m) flux, numu_cv / nue_cv'),
               neut=neut, neut_scan=scan)
    os.makedirs(paths.INPUTS_DIR, exist_ok=True)
    json.dump(out, open(PRED, 'w'), indent=0)
    print(f'wrote {PRED}')


# ===================================================================================================== step 2
def load_budget(path):
    R = json.load(open(path))
    if 'total' in R:                       # results/07_budget.json (v3)
        return dict(sigma=R['sigma'], stat=R['stat'], up=R['total'][0], down=R['total'][1], genie=R['sigma_genie'],
                    base=R['sigma_base'])
    return dict(sigma=R['sigma'], stat=R['stat'], up=R['total_up'], down=R['total_down'],   # pre-v3 budget JSON
                genie=R['sigma_genie'], base=R['per_model']['Base']['sigma'])


def draw(out, budget_path, placement, column):
    P = json.load(open(PRED))
    B = load_budget(budget_path)
    t2k = dict(T2K)
    font = {'family': 'serif', 'size': 8 if column else 12}
    mpl.rc('font', **font)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['figure.figsize'] = (5, 4)

    gsp = P['genie_spline']
    GeV_vals = np.asarray(gsp['E_p'])
    total_nu_O16_xs = np.asarray(gsp['xs_p']) + np.interp(GeV_vals, gsp['E_n'], gsp['xs_n'])
    fl_mu, fl_e, edges = (np.asarray(P['fv_flux'][k]) for k in ('numu', 'nue', 'edges'))
    total_flux = fl_mu / 1e6 + fl_e / 1e6
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(total_flux); tot = total_flux.sum()
    median_energy = bin_centers[np.argmax(cum >= tot / 2)]
    E_lower = bin_centers[np.argmax(cum >= tot * 0.16)]
    E_upper = bin_centers[np.argmax(cum >= tot * 0.84)]
    mean_energy = np.sum(bin_centers * total_flux) / tot
    below, above = bin_centers < mean_energy, bin_centers >= mean_energy
    sd_lo = np.sqrt(np.sum(total_flux[below] * (bin_centers[below] - mean_energy) ** 2) / total_flux[below].sum())
    sd_hi = np.sqrt(np.sum(total_flux[above] * (bin_centers[above] - mean_energy) ** 2) / total_flux[above].sum())
    x_pt, x_err = mean_energy, [[sd_lo], [sd_hi]]
    if placement == 'median':
        s_bnb = summary(edges, fl_mu + fl_e)
        x_pt, x_err = s_bnb['median'], [[s_bnb['median'] - s_bnb['q16']], [s_bnb['q84'] - s_bnb['median']]]
        s_t2k = summary(*read_t2k('t2kflux_2020_plus250kA_runcond_sk.txt')[:2])
        t2k.update(x=s_t2k['median'], x_lo=s_t2k['q16'], x_hi=s_t2k['q84'])
    xs_spline_avg = np.sum(np.interp(bin_centers, GeV_vals, total_nu_O16_xs) * total_flux) / tot
    nt = P['neut']
    neut_gs = sum(r['gs'] * r['phi_gs'] for r in nt.values()) / sum(r['phi_gs'] for r in nt.values())
    neut_fv = sum(r['fv'] * r['phi_fv'] for r in nt.values()) / sum(r['phi_fv'] for r in nt.values())
    neut_sigma_combined = neut_gs
    Es = np.array([s['E'] for s in P['neut_scan']]); sig = np.array([s['sigma'] for s in P['neut_scan']])
    sm = savgol_filter(sig, window_length=7, polyorder=2)
    spl = PchipInterpolator(np.concatenate(([0.0], Es)), np.concatenate(([0.0], sm)))
    neut_E_fine = np.linspace(0.0, Es.max(), 400)
    neut_sigma_fine = spl(neut_E_fine)
    GENIE_FLUX_AVG = B['genie']
    data_val = B['sigma']
    stat_abs = data_val * B['stat'] / 100
    err_up, err_down = data_val * B['up'] / 100, data_val * B['down'] / 100
    deexc_alt_val = B['base']
    print(f'placement {placement}: ANNIE x = {x_pt:.3f}, T2K x = {t2k["x"]:.3f}; GENIE {GENIE_FLUX_AVG:.4f} (spline x FV flux '
          f'{xs_spline_avg:.4f}); NEUT {neut_gs / 1e-38:.4f} (FV-flux reweighted {neut_fv / 1e-38:.4f})')
    print(f'ANNIE: {data_val:.3f} +/- {stat_abs:.3f} (stat), total +{err_up:.3f} -{err_down:.3f}; base GENIE de-exc. {deexc_alt_val:.3f}')

    # ---------------- plot (styling unchanged from the PRD draft-1 figure)
    K = 0.65 if column else 1.0
    fig, ax = plt.subplots(figsize=(3.375, 3.25) if column else (7, 4.5))
    ax2 = ax.twinx()
    flux_hist = ax2.hist(bin_centers, bins=len(bin_centers), weights=total_flux, color='black', alpha=0.1, zorder=1,
                         label='BNB ' + r'$\nu$' + ' Flux')
    ax2.set_ylabel(''); ax2.set_yticks([]); ax2.spines['right'].set_visible(False)
    genie_line, = ax.plot(GeV_vals, total_nu_O16_xs, color='#0066ff', linewidth=1, linestyle='dashed', label='GENIE 3.0.6', zorder=9)
    genie_pt = ax.errorbar(x_pt, GENIE_FLUX_AVG, xerr=x_err, fmt='o', color='#0066ff', ecolor='#0066ff', markersize=5 * K,
                           elinewidth=2 * K, capsize=4 * K, alpha=0.85, zorder=10, label='GENIE flux-averaged')
    neut_line, = ax.plot(neut_E_fine, neut_sigma_fine / 1e-38, color='#cc3333', linestyle='dotted', linewidth=1.5, zorder=8, label='NEUT 6.1.4')
    neut_pt = ax.errorbar(x_pt, neut_sigma_combined / 1e-38, xerr=x_err, fmt='s', color='#cc3333', ecolor='#cc3333',
                          markersize=5 * K, elinewidth=2 * K, capsize=4 * K, alpha=0.85, zorder=10, label='NEUT flux-averaged')
    ax.errorbar(x_pt, data_val, xerr=x_err, yerr=stat_abs, fmt='ko', markersize=6 * K, elinewidth=1.5 * K, capsize=5 * K, zorder=99)
    data_pt = ax.errorbar(x_pt, data_val, yerr=[[err_down], [err_up]], fmt='none', ecolor='k', elinewidth=1 * max(K, 0.8),
                          capsize=5 * K, zorder=11, label='ANNIE ' + r'$2.57 \times 10^{20}$' + ' POT')
    deexc_pt, = ax.plot(x_pt, deexc_alt_val, marker='D', markerfacecolor='none', markeredgecolor='cyan',
                        markeredgewidth=1.2 * max(K, 0.8), markersize=8 * K, linestyle='none', zorder=98, label='Base GENIE de-excitation')
    ax.errorbar(t2k['x'], t2k['val'], xerr=[[t2k['x'] - t2k['x_lo']], [t2k['x_hi'] - t2k['x']]], yerr=t2k['stat'], fmt='^',
                color='#2ca02c', ecolor='#2ca02c', markersize=6 * K, elinewidth=1.2 * K, capsize=4 * K, zorder=13)
    t2k_pt = ax.errorbar(t2k['x'], t2k['val'], yerr=[[t2k['err_down']], [t2k['err_up']]], fmt='^', color='#2ca02c',
                         ecolor='#2ca02c', markersize=6 * K, elinewidth=1 * max(K, 0.8), capsize=4 * K, alpha=0.85, zorder=11,
                         label='T2K ' + r'$14.94 \times 10^{20}$' + ' POT')
    ax.set_xlabel(r'$E_\nu$' + ' [GeV]', loc='right', fontsize=9 if column else 14)
    ax.set_ylabel(r'$\sigma_{\text{NCQE}}$' + ' [' + r'$10^{-38} \text{cm}^{2}$' + ']', loc='top', fontsize=9 if column else 14)
    ax.set_xlim([0, 3]); ax.set_ylim([0, 5])
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_xlim(ax.get_xlim())
    ax.legend(handles=[genie_line, genie_pt, neut_line, neut_pt, deexc_pt, data_pt, t2k_pt, flux_hist[2][0]],
              **(dict(fontsize=5.6, loc='lower right', bbox_to_anchor=(1.0, 0.165), frameon=False, borderaxespad=0.2,
                      handlelength=1.6, labelspacing=0.22, borderpad=0.2) if column else
                 dict(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.02))))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.03 if column else .3, facecolor='w')
    print('saved', out)
    vals = dict(placement=placement, plotted_x=x_pt, x_err=x_err, x_semidev=(sd_lo, sd_hi), median_energy=median_energy,
                E_lower=E_lower, E_upper=E_upper, mean_energy=mean_energy, genie_flux_avg=GENIE_FLUX_AVG,
                genie_spline_x_fv=xs_spline_avg, neut_gs=neut_gs / 1e-38, neut_fv=neut_fv / 1e-38, data=data_val,
                stat=stat_abs, err_up=err_up, err_down=err_down, base=deexc_alt_val, t2k=t2k,
                provenance=provenance(dict(script='09_xs_money_plot.py', budget=os.path.abspath(budget_path),
                                           predictions=PRED, predictions_md5=md5(PRED), layout='column' if column else 'wide')))
    json.dump(vals, open(os.path.splitext(out)[0] + '_values.json', 'w'), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default=os.path.join(paths.REPO, 'plots', 'NCQE_comparison_median_column.png'))
    ap.add_argument('--budget', default=os.path.join(paths.RESULTS_DIR, '07_budget.json'))
    ap.add_argument('--placement', choices=('median', 'mean'), default='median')
    ap.add_argument('--layout', choices=('column', 'wide'), default='column')
    ap.add_argument('--rebuild-inputs', action='store_true')
    a = ap.parse_args()
    if a.rebuild_inputs or not os.path.exists(PRED):
        rebuild_inputs()
    draw(a.out, a.budget, a.placement, a.layout == 'column')


if __name__ == '__main__':
    main()
