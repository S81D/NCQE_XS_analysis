# Pass over the 44-knob GENIE + 13-knob flux reweighting production (v3; used by 08_systematics.py).
#
# The production (paths.MC_44KNOB_DIR) contains the same GENIE/WCSim events as the CV sample MC_GENIE-O16, file by
# file and entry by entry, but its spill timing and reconstruction were re-randomised when it was reprocessed. Two
# evaluations are therefore possible and both are provided:
#   ratio transfer (defines the paper's interaction-model and flux terms): the CV selection is re-run on the production
#       and each category's universe/CV ratio, measured there, is applied to the frozen CV yields;
#   event level (cross-check): the universe weights are read at the trigger entries of the CV-selected events,
#       after checking that the truth record of the two files is identical entry by entry.
# Weight conventions (MicroBooNE xsec_analyzer, UniverseMaker): GENIE weights (All0 multisim, unisim slots) are used
# as stored (they already contain the tuned CV); flux weights are multiplied by the TunedCV weight; safe(): non-finite,
# negative or > 30 -> 1.
import os
import numpy as np
import uproot
import awkward as ak

from . import paths
from .selection import mc_select_file, truth_masks, cv_weight

FLUX_KNOBS = ['horncurrent_FluxUnisim', 'expskin_FluxUnisim', 'piplus_PrimaryHadronSWCentralSplineVariation',
              'piminus_PrimaryHadronSWCentralSplineVariation', 'kplus_PrimaryHadronFeynmanScaling',
              'kzero_PrimaryHadronSanfordWang', 'kminus_PrimaryHadronNormalization', 'pioninexsec_FluxUnisim',
              'pionqexsec_FluxUnisim', 'piontotxsec_FluxUnisim', 'nucleoninexsec_FluxUnisim',
              'nucleonqexsec_FluxUnisim', 'nucleontotxsec_FluxUnisim']
# Unisim dial -> output branch slot. PhaseIITreeMaker writes fixed branch names, so the reprocessing config
# (LoadReweightGenieEventConfig_NCQE, 2026-10-12) re-used existing calculator names. Universe 0 / 1 =
#   NormNCMEC +100% / -100%; MaNCEL, EtaNCEL, MaCCQE, CoulombCCQE, RPA_CCQE +1 / -1 sigma; Ax/VecFFCCQEshape 1 / 0.
UNISIM_SLOTS = {'NormNCMEC': 'XSecShape_CCMEC', 'MaNCEL': 'DecayAngMEC', 'EtaNCEL': 'Theta_Delta2Npi',
                'MaCCQE': 'NormCCCOH', 'CoulombCCQE': 'ThetaDelta2NRad', 'RPA_CCQE': 'RPA_CCQE',
                'AxFFCCQEshape': 'AxFFCCQEshape', 'VecFFCCQEshape': 'VecFFCCQEshape'}
TRUTH_KEYS = ['trueNeutrinoEnergy', 'trueNuIntxVtx_X', 'trueNuIntxVtx_Y', 'trueNuIntxVtx_Z', 'trueNuPDG', 'trueFSLPdg']


def safe(w):
    w = np.array(w, dtype=np.float64)
    w[~np.isfinite(w) | (w < 0) | (w > 30)] = 1.0
    return w


def _jag(S, name, width):
    a = S[name].array()
    return ak.to_numpy(ak.fill_none(ak.pad_none(a, width, clip=True), np.nan)).astype(np.float64).reshape(-1, width)


def reweight_file(args):
    """args = (file name, CV-selected trigger entries of the same-named nominal file (in-window events, any bunch))."""
    name, nom_ce = args
    out = dict(name=name, ok=False, matched=False)
    p_new, p_nom = paths.MC_44KNOB_DIR + name, paths.MC_EMPIRICAL_DIR + name
    try:
        sel, truth = mc_select_file(p_new)
        root = uproot.open(p_new)
        S = root['phaseIITriggerTree']
    except Exception as e:                                  # empty / truncated files of the production
        out['why'] = f'unreadable: {type(e).__name__}'
        return out
    with root:
        tm = truth_masks(S)
        w_ev, cv_raw = cv_weight(S)
        a0 = _jag(S, 'weight_All0_UBGenie', 100)
        kn = np.stack([_jag(S, f'weight_{k}', 100) for k in FLUX_KNOBS])
        un = np.stack([_jag(S, f'weight_{s}_UBGenie', 2) for s in UNISIM_SLOTS.values()])
        tnew = {k: S[k].array(library='np') for k in TRUTH_KEYS}
        n = S.num_entries
    out['ok'] = True
    out['n_bad'] = int((~np.isfinite(a0)).all(1).sum() + (~np.isfinite(kn)).all(2).sum() + (~np.isfinite(un)).all(2).sum())
    cvr = np.where(np.isfinite(cv_raw), cv_raw, np.nan)

    def wsets(ix):
        return dict(a0=a0[ix], kn=kn[:, ix], un=un[:, ix], cvraw=cv_raw[ix])

    # truth denominators: true FV nuNCQE
    sig = np.where(tm['fv'] & tm['nu'])[0]
    out['T'] = dict(cv=float(w_ev[sig].sum()), cv2=float((w_ev[sig] ** 2).sum()), n=int(len(sig)),
                    x=safe(a0[sig]).sum(0), f=safe(np.prod(kn[:, sig], axis=0) * cvr[sig, None]).sum(0),
                    k=np.stack([safe(kn[i, sig] * cvr[sig, None]).sum(0) for i in range(len(FLUX_KNOBS))]),
                    u=safe(un[:, sig]).sum(1))
    # ratio transfer: re-selected in-window events of the production
    if sel is not None:
        iw = sel['in_win']
        out['sel'] = {k: v[iw] for k, v in sel.items()}
        out['sel'].update(wsets(sel['ce'][iw]))
    # event level: weights at the nominal CV entries, if the truth record is identical
    if nom_ce is not None and os.path.exists(p_nom):
        with uproot.open(p_nom) as fn:
            Sn = fn['phaseIITriggerTree']
            same = Sn.num_entries == n and all(np.array_equal(Sn[k].array(library='np'), tnew[k]) for k in TRUTH_KEYS)
            same &= np.array_equal(cv_weight(Sn)[1], cv_raw, equal_nan=True)
        out['matched'] = bool(same)
        if same:
            out['nom'] = wsets(nom_ce)
    return out
