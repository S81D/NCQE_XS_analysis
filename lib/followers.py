# Follower-particle records for the primary de-excitation gamma study (06, v3).
#
# Reads, for every MC file, the follower arrays of (a) the CV-selected events passed in and (b) every true FV nuNCQE
# event (population check against paper Table II). Follower start times are returned relative to the true interaction
# time, as in v2 notebook 06 (MC_FollowerT - MC_Vtt).
import os
import numpy as np
import uproot
import awkward as ak
from concurrent.futures import ProcessPoolExecutor

from . import paths
from .selection import truth_masks

BRANCHES = ['trueFollowerE', 'trueFollowerPDG', 'trueFollowerParentPDG', 'trueFollowerStartT']


def _read(args):
    path, entries = args
    with uproot.open(path) as f:
        S = f['phaseIITriggerTree']
        tm = truth_masks(S)
        fv_sig = np.where(tm['fv'] & tm['nu'])[0]
        need = np.union1d(fv_sig, entries).astype(np.int64)
        a = {b: S[b].array()[need] for b in BRANCHES}
        vt = S['trueNuIntxVtx_T'].array(library='np')[need]
        nuc = S['trueTargetNucleonPDG'].array(library='np')[need]
    return dict(entries=need, E=a['trueFollowerE'], PDG=a['trueFollowerPDG'], parent=a['trueFollowerParentPDG'],
                T=a['trueFollowerStartT'] - vt, nucleon=nuc, truth_fv=np.isin(need, fv_sig))


def read_followers(directory, names, entries_by_file):
    """entries_by_file: {file index: trigger entries}. Returns one record per (file, entry) with keys
    fidx, entry, E, PDG, parent, T (awkward), nucleon, truth_fv."""
    jobs = [(os.path.join(directory, n), entries_by_file.get(i, np.zeros(0, np.int64))) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=paths.N_WORKERS) as ex:
        res = list(ex.map(_read, jobs, chunksize=20))
    out = dict(fidx=np.concatenate([np.full(len(r['entries']), i, np.int32) for i, r in enumerate(res)]),
               entry=np.concatenate([r['entries'] for r in res]),
               nucleon=np.concatenate([r['nucleon'] for r in res]),
               truth_fv=np.concatenate([r['truth_fv'] for r in res]))
    for k in ('E', 'PDG', 'parent', 'T'):
        out[k] = ak.concatenate([r[k] for r in res])
    return out
