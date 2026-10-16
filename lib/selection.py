# Vectorised nuNCQE event selection (v3).
#
# Same cuts, inequalities and float behaviour as 05_FULL_DATA_signal_band.ipynb (cluster loops); validated to reproduce
# the notebook-05 yields exactly (MC per category, data, off-beam). The MC selection is run WITHOUT the +-1 sigma bunch
# cut and the first-26-bunch cut, which are returned as flags (in_win, in26), so that the same pass also provides the
# bunch-timing sidebands. The CV selection of the paper is in_win & in26.
#
# Functions used by the worker pools are defined at module level so that multiprocessing also works with the 'spawn'
# start method (macOS default), including when called from a notebook.
import os
import numpy as np
import pandas as pd
import uproot
import awkward as ak
from concurrent.futures import ProcessPoolExecutor

from .constants import (FV_radius, FV_half_height, radius, half_height, bunch_sigma, bunch_offset_mapping, centers,
                        centers_data, bunch_time_start, bunch_time_end, bunch_time_start_data, bunch_time_end_data,
                        MC_third_spill_cut, data_third_spill_cut, time_to_prompt_end)
from .reco_functions import reco_energy, pairwise_relative_direction, is_inside_rotated_parabola
from . import paths

# ---------------------------------------------------------------------------------------------------- cut helpers
PREBEAM = (0.0, 150.0)            # BRF-corrected cluster time window [ns] before the first bunch (beam-free, see 03)
LIVE_SIG = 26 * 2 * bunch_sigma   # 182 ns: 26 bunch windows of +-1 sigma
DATA_BRANCHES = ['run_number', 'cluster_time', 'cluster_time_BRF', 'cluster_PE', 'cluster_Qb', 'cluster_Hits',
                 'recoVtx', 'recoVty', 'recoVtz', 'TankMRDCoinc', 'NoVeto', 'only_prompt_cluster',
                 'MRD_activity', 'FMV_activity', 'part_file', 'event_number', 'cluster_Number']
CAT_NAMES = {0: 'nuNCQE (FV)', 1: 'NCother', 2: 'CC', 3: 'nubarNCQE', 4: 'OOFV', 5: 'External'}


def core_cuts(qb, rvx, rvy, rvz, pe):
    """q_b < 0.6, vertex found (>= -50), reco FV, 5 <= E_reco <= 12 MeV. Written with ~(reject) so NaN passes, as in 05."""
    r2 = rvx ** 2 + rvz ** 2
    E = reco_energy(pe)
    return (~(qb >= 0.6) & ~(rvx < -50) & ~(r2 > FV_radius ** 2) & ~(np.abs(rvy) > FV_half_height)
            & ~(E < 5) & ~(E > 12))


def in_bunch_window(t, cen, offset=bunch_offset_mapping):
    """lib.reco_functions.bunch_cutting vectorised: |(centre - offset) - t| < sigma for any bunch."""
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros(len(t), dtype=bool)
    shifted = cen - offset
    for lo in range(0, len(t), 200000):
        tt = t[lo:lo + 200000]
        out[lo:lo + 200000] = (np.abs(shifted[None, :] - tt[:, None]) < bunch_sigma).any(axis=1)
    return out


def signal_window_data(t, offset=bunch_offset_mapping):
    """Data signal window: spill start .. first-26-bunch cut, and within +-1 sigma of a predicted nuNCQE centre."""
    start = centers_data[0] - offset - bunch_sigma
    third = centers_data[25] - offset + bunch_sigma
    ok = ~(t < start) & ~(t > bunch_time_end_data) & ~(t > third)
    idx = np.where(ok)[0]
    ok[idx] = in_bunch_window(t[idx], centers_data, offset)
    return ok


def signed_dt(t, cen, offset=bunch_offset_mapping):
    """(predicted centre) - t for the nearest bunch (positive = cluster earlier than predicted)."""
    sh = cen - offset
    t = np.asarray(t, float)
    out = np.empty(len(t))
    for lo in range(0, len(t), 100000):
        d = sh[None, :] - t[lo:lo + 100000, None]
        out[lo:lo + 100000] = d[np.arange(len(d)), np.abs(d).argmin(axis=1)]
    return out


def abs_dt(t, cen, offset=bunch_offset_mapping):
    return np.abs(signed_dt(t, cen, offset))


def direction_ok(hx, hy, hz, hpe, ht):
    """Charge-coherence (C_V) cut of 05: keep clusters outside the rotated parabola."""
    cw = pairwise_relative_direction(hx, hy, hz, hpe, ht)
    return bool(is_inside_rotated_parabola(cw[2], cw[0]))


def _categories(ext, fv, nc, qel, z, pdg):
    """Notebook-05 categories: 0 nuNCQE (FV), 1 NCother, 2 CC, 3 nubarNCQE, 4 OOFV, 5 External; -1 unassigned."""
    cat = np.full(len(ext), -1, dtype=np.int8)
    cat[ext] = 5
    f_ = ~ext & fv
    cat[f_ & (nc != 1)] = 2
    cat[f_ & (nc == 1) & (qel != 1)] = 1
    cat[f_ & (nc == 1) & (qel == 1) & (z != 8)] = 1
    fo = f_ & (nc == 1) & (qel == 1) & (z == 8)
    cat[fo & ((pdg == 14) | (pdg == 12))] = 0
    cat[fo & ((pdg == -14) | (pdg == -12))] = 3
    cat[~ext & ~fv] = 4
    return cat


# ---------------------------------------------------------------------------------------------------- MC
def truth_masks(S):
    """True-vertex FV / External masks and nuNCQE (16O, nu_mu + nu_e) flag per trigger (05 coordinate transform)."""
    g = lambda b: S[b].array(library='np')
    vx, vy, vz = g('trueNuIntxVtx_X') / 100, g('trueNuIntxVtx_Y') / 100 + 0.1446, g('trueNuIntxVtx_Z') / 100 - 1.681
    r2t = vx ** 2 + vz ** 2
    ext = (r2t > radius ** 2) | (np.abs(vy) > half_height)
    fv = (r2t < FV_radius ** 2) & (np.abs(vy) < FV_half_height)
    NC, QEL, Z, PDG = g('trueNC'), g('trueQEL'), g('trueTargetZ'), g('trueFSLPdg')
    nu = (NC == 1) & (QEL == 1) & (Z == 8) & ((PDG == 14) | (PDG == 12))
    nubar = (NC == 1) & (QEL == 1) & (Z == 8) & ((PDG == -14) | (PDG == -12))
    return dict(ext=ext, fv=fv, NC=NC, QEL=QEL, Z=Z, PDG=PDG, nu=nu, nubar=nubar)


def cv_weight(S):
    """TunedCV weight per trigger: NaN if missing, +-inf -> 1 (notebook convention)."""
    w = ak.to_numpy(ak.fill_none(ak.firsts(S['weight_TunedCentralValue_UBGenie'].array()), np.nan)).astype(float)
    raw = w.copy()
    w[np.isinf(w)] = 1.0
    return w, raw


def mc_select_file(path):
    """All cuts of 05 except the +-1 sigma bunch cut and the first-26-bunch cut (returned as flags in_win, in26).

    Returns (sel, truth): sel = per selected cluster (None if no cluster tree / nothing selected), truth = weighted and
    raw counts of true FV nuNCQE and in-tank OOFV nuNCQE, plus the number of triggers."""
    with uproot.open(path) as root:
        T, S = root['phaseIITankClusterTree'], root['phaseIITriggerTree']
        tm = truth_masks(S)
        w_ev, _ = cv_weight(S)
        sig_t = tm['fv'] & tm['nu']
        oofv_t = ~tm['ext'] & ~tm['fv'] & tm['nu']
        truth = np.array([w_ev[sig_t].sum(), sig_t.sum(), w_ev[oofv_t].sum(), oofv_t.sum(), S.num_entries,
                          (w_ev[sig_t] ** 2).sum()], dtype=float)
        if T.num_entries == 0:
            return None, truth
        a = T.arrays(['eventNumber', 'clusterNumber', 'clusterPE', 'clusterChargeBalance', 'clusterHits', 'bunchTimes',
                      'TankMRDCoinc', 'NoVeto', 'recoLeastSqVtxX', 'recoLeastSqVtxY', 'recoLeastSqVtxZ'], library='np')
        CEN, cn, bt = a['eventNumber'], a['clusterNumber'], a['bunchTimes']
        mrd, fmv = S['MRDhitT'].array(), S['FMVhitT'].array()
        mrd_any = ak.to_numpy(ak.any((mrd >= 0) & (mrd <= 300), axis=1))
        fmv_any = ak.to_numpy(ak.any((fmv >= 0) & (fmv <= 300), axis=1))
        m = (cn == 0) & ~(bt > time_to_prompt_end) & ~(a['clusterHits'] < 10)                 # prompt cluster
        m &= ~((a['TankMRDCoinc'] == 1) | (a['NoVeto'] == 0) | mrd_any[CEN] | fmv_any[CEN])    # no FMV / MRD activity
        m &= ~(bt > bunch_time_end) & ~(bt < bunch_time_start)                                 # in spill
        seg = np.cumsum(cn == 0)                                                               # single prompt cluster
        other = (cn != 0) & (bt < time_to_prompt_end)
        m &= ~(np.bincount(seg, weights=other.astype(float), minlength=seg.max() + 1) > 0)[seg]
        m &= core_cuts(a['clusterChargeBalance'], a['recoLeastSqVtxX'], a['recoLeastSqVtxY'], a['recoLeastSqVtxZ'],
                       a['clusterPE'])                                                          # q_b, FV, E_reco
        idx = np.where(m)[0]
        if len(idx):                                                                           # C_V
            hx, hy, hz, hp, ht = (ak.to_list(T[b].array()[idx]) for b in ('hitX', 'hitY', 'hitZ', 'hitPE', 'hitT'))
            idx = idx[np.array([direction_ok(*h) for h in zip(hx, hy, hz, hp, ht)], dtype=bool)]
        if len(idx) == 0:
            return None, truth
        ce = CEN[idx]
        cat = _categories(tm['ext'][ce], tm['fv'][ce], tm['NC'][ce], tm['QEL'][ce], tm['Z'][ce], tm['PDG'][ce])
        t = bt[idx].astype(np.float64)
        sel = dict(ce=ce.astype(np.int64), t=t, in_win=in_bunch_window(t, centers), in26=~(t > MC_third_spill_cut),
                   cat=cat, w=w_ev[ce], oofv_nu=(cat == 4) & tm['nu'][ce], oofv_nubar=(cat == 4) & tm['nubar'][ce])
        return sel, truth


def _mc_chunk(paths_):
    out, truth = [], np.zeros(6)
    for i, p in paths_:
        s, t = mc_select_file(p)
        truth += t
        if s is not None:
            s['fidx'] = np.full(len(s['ce']), i, dtype=np.int32)   # index into the sorted file names
            out.append(s)
    sel = {k: np.concatenate([s[k] for s in out]) for k in out[0]} if out else {}
    return sel, truth


def mc_selection(tag, directory, refresh=False, log=print):
    """Selection over every *.root file of `directory` (sorted names), cached in cache/mc_sel_<tag>.npz."""
    os.makedirs(paths.CACHE_DIR, exist_ok=True)
    cache = os.path.join(paths.CACHE_DIR, f'mc_sel_{tag}.npz')
    names = sorted(f for f in os.listdir(directory) if f.endswith('.root'))
    if os.path.exists(cache) and not refresh:
        z = dict(np.load(cache, allow_pickle=False))
        if list(z['files']) == names and len(z['truth']) == 6:
            log(f'  MC {tag}: selection from cache ({len(names)} files)')
            return z
    log(f'  MC {tag}: selecting {len(names)} files in {directory} ...')
    jobs = [[(i, os.path.join(directory, n)) for i, n in enumerate(names)][k:k + 25] for k in range(0, len(names), 25)]
    with ProcessPoolExecutor(max_workers=paths.N_WORKERS) as ex:
        res = list(ex.map(_mc_chunk, jobs))
    sel = {k: np.concatenate([r[0][k] for r in res if r[0]]) for k in next(r[0] for r in res if r[0])}
    sel['truth'] = np.sum([r[1] for r in res], axis=0)  # [FV nu w, FV nu raw, OOFV nu w, OOFV nu raw, triggers, FV nu w^2]
    sel['files'] = np.array(names)
    np.savez(cache, **sel)
    return sel


def cv_mask(sel):
    """The paper's CV selection: +-1 sigma bunch window and first 26 bunches."""
    return sel['in_win'] & sel['in26']


# ---------------------------------------------------------------------------------------------------- data
def load_ntuples(directory, branches=DATA_BRANCHES):
    files = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.root'))
    parts, fidx, entry = [], [], []
    for i, p in enumerate(files):
        with uproot.open(p) as f:
            a = f['data'].arrays(branches, library='np')
        parts.append(a)
        fidx.append(np.full(len(a['run_number']), i, dtype=np.int32))
        entry.append(np.arange(len(a['run_number']), dtype=np.int64))
    d = {k: np.concatenate([p[k] for p in parts]) for k in branches}
    d['file_idx'] = np.concatenate(fidx)
    d['entry'] = np.concatenate(entry)
    return d, files


def _dir_worker(args):
    path, entries, gidx = args
    with uproot.open(path) as f:
        t = f['data']
        hx, hy, hz, hp, ht = (ak.to_list(t[b].array()[entries]) for b in ('hitX', 'hitY', 'hitZ', 'hitPE', 'hitT'))
    return gidx, np.array([direction_ok(*h) for h in zip(hx, hy, hz, hp, ht)], dtype=bool)


def compute_direction(d, files, cand, tag, refresh=False, log=print):
    """C_V cut for the clusters flagged in `cand` (hit-level, slow), cached in cache/dir_<tag>.npz."""
    os.makedirs(paths.CACHE_DIR, exist_ok=True)
    cache = os.path.join(paths.CACHE_DIR, f'dir_{tag}.npz')
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        if z['cand'].shape == cand.shape and (z['cand'] == cand).all():
            log(f'  [{tag}] direction cut from cache ({cand.sum()} candidates)')
            return z['dir_ok']
    gl = np.where(cand)[0]
    jobs = [(files[fi], d['entry'][gl[d['file_idx'][gl] == fi]], gl[d['file_idx'][gl] == fi])
            for fi in np.unique(d['file_idx'][gl])]
    log(f'  [{tag}] direction cut for {len(gl)} clusters in {len(jobs)} files ...')
    dir_ok = np.zeros(len(cand), dtype=bool)
    with ProcessPoolExecutor(max_workers=paths.N_WORKERS) as ex:
        for g, ok in ex.map(_dir_worker, jobs, chunksize=4):
            dir_ok[g] = ok
    np.savez(cache, cand=cand, dir_ok=dir_ok)
    return dir_ok


def data_pot(directory=None):
    """Triggers and POT (TOR860, 1e20) of the on-beam runs present in `directory` (05 / 07 convention)."""
    directory = directory or paths.DATA_DIR
    runs = [int(f.split('_')[0][1:]) for f in os.listdir(directory) if f.endswith('.root')]
    df = pd.read_csv(paths.POT_CSV)
    df = df[df['RUN'].isin(runs)]
    return int(df['TOTAL_TRIGGERS'].sum()), float(df['TOR860_POTe12'].sum() / 1e8)


def data_selection(refresh=False, log=print):
    """On-beam data with every non-timing cut of 05 (single prompt cluster, veto, q_b, FV, E_reco, C_V).

    Returns the BRF-corrected cluster times of the selected clusters (`t`), the signal count N_obs, the pre-beam
    count (CIT estimate, 03 convention) and the spill sample used by the skyshine and bunch-timing studies."""
    on, files = load_ntuples(paths.DATA_DIR)
    tB = on['cluster_time_BRF'].astype(np.float64)
    core = core_cuts(on['cluster_Qb'], on['recoVtx'], on['recoVty'], on['recoVtz'], on['cluster_PE'])
    spill = ~(tB < bunch_time_start_data) & ~(tB > bunch_time_end_data)
    cand = core & (((tB >= -5) & (tB <= 700)) | spill)
    dirok = compute_direction(on, files, cand, 'onbeam', refresh, log)
    veto = (on['TankMRDCoinc'] != 1) & (on['NoVeto'] != 0) & (on['MRD_activity'] == 0) & (on['FMV_activity'] == 0)
    base = (on['only_prompt_cluster'] == 1) & veto & core & dirok
    lo, hi = PREBEAM
    out = dict(t=tB[base], n_obs=int((base & signal_window_data(tB)).sum()),
               pre_raw=int((base & (tB >= lo) & (tB < hi)).sum()), t_spill=tB[base & spill],
               n_files=len(files), n_clusters=len(tB))
    out['trig'], out['pot'] = data_pot()
    return out
