import awkward as ak
import numpy as np
import numba as nb
from coffea.nanoevents.methods import vector

@nb.njit
def inner_dim_indexing(target, index, builder):
    for t, idx in zip(target, index):
        builder.begin_list()
        if -1 not in idx and 99 not in idx:
            for i in idx:
                builder.real(t[i])
        builder.end_list()
    return builder

def clean_sv(df):
    # Leptons
    pass_fiducial = (df.lep_Hindex[:,2] != -1) & (df.lep_Hindex[:,2] != 99) & (ak.num(df.lep_pt) == ak.num(df.lep_id)) # not sure why the length is different..

    _df = df.mask[pass_fiducial]
    _lep_unselected = ak.zip(dict(
        pt=_df.lep_pt,
        pt_reg=np.log(_df.lep_pt + 1),
        eta=_df.lep_eta,
        phi=_df.lep_phi,
        mass=_df.lep_mass,
        mass_reg=np.log(_df.lep_mass + 1),
        isEle=abs(_df.lep_id) == 11,
        isMuon=abs(_df.lep_id) == 13,
        isTau=abs(_df.lep_id) == 15,
        isPosCharged=_df.lep_id > 0,
        ),
    )
    _lep = {}
    for b in ak.fields(_lep_unselected):
        _lep[b] = inner_dim_indexing(ak.values_astype(_lep_unselected[b], np.float32), df.lep_Hindex, ak.ArrayBuilder()).snapshot()
    _lep = ak.zip(_lep)

    _lep_lo = ak.zip(
        dict(
            pt=_lep.pt,
            eta=_lep.eta,
            phi=_lep.phi,
            mass=_lep.mass,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )

    ## SVs
    sv = ak.zip(dict(
        pt=df.sv_pt,
        pt_reg=np.log(df.sv_pt + 1),
        eta=df.sv_eta,
        phi=df.sv_phi,
        mass=df.sv_mass,
        mass_reg=np.log(df.sv_mass + 1),
        ParticleNet_b=df.sv_ParticleNet_b,
        ParticleNet_bb=df.sv_ParticleNet_bb,
        ParticleNet_c=df.sv_ParticleNet_c,
        ParticleNet_cc=df.sv_ParticleNet_cc,
        ParticleNet_unmat=df.sv_ParticleNet_unmat,
        ),
    )
    sv_lo = ak.zip(
        dict(
            pt=sv.pt,
            eta=sv.eta,
            phi=sv.phi,
            mass=sv.mass,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    sv['px'] = sv_lo.x
    sv['py'] = sv_lo.y
    sv['pz'] = sv_lo.z
    sv['e'] = sv_lo.t

    def check_clean_from_obj(target, cleanded_obj):
        comb = ak.cartesian([target, cleanded_obj], nested=True)
        dr = comb['0'].delta_r(comb['1'])
        return ak.all(dr > 0.4, axis=-1)

    # clean from all tight leptons
    isoCutMu, isoCutEl = 0.35, 9999.
    passed_idiso = ((abs(df.lep_id)==13) & (df.lep_RelIso<=isoCutMu)) | ((abs(df.lep_id)==11) & (df.lep_RelIso<=isoCutEl))
    _lep_unselected_passed_idiso = _lep_unselected[passed_idiso]
    tightleps_unselected_lo = ak.zip(
        dict(
            pt=_lep_unselected_passed_idiso.pt,
            eta=_lep_unselected_passed_idiso.eta,
            phi=_lep_unselected_passed_idiso.phi,
            mass=_lep_unselected_passed_idiso.mass,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    sv_is_iso_all_tightleps = check_clean_from_obj(sv_lo, tightleps_unselected_lo)

    # clean from all higgs cands leptons
    sv_is_iso_all_candleps = check_clean_from_obj(sv_lo, _lep_lo)

    # clean from all FSR photons matched with tight leps
    fsrph_lo = ak.zip(
        dict(
            pt=df.fsrPhotons_pt,
            eta=df.fsrPhotons_eta,
            phi=df.fsrPhotons_phi,
            mass=ak.zeros_like(df.fsrPhotons_pt),
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    _fsrph_matched_lep_tightid = df.lep_tightId[df.fsrPhotons_lepindex]
    _fsrph_matched_lep_iso = df.lep_RelIsoNoFSR[df.fsrPhotons_lepindex]
    _fsrph_matched_lep_id = df.lep_id[df.fsrPhotons_lepindex]
    fsrph_is_qual = (_fsrph_matched_lep_tightid==1) & ((abs(_fsrph_matched_lep_id)==13) & (_fsrph_matched_lep_iso<=isoCutMu)) | ((abs(_fsrph_matched_lep_id)==11) & (_fsrph_matched_lep_iso<=isoCutEl))
    fsrph_qual_lo = fsrph_lo.mask[pass_fiducial][fsrph_is_qual] # should pass fiducial because fsrPhotons_pt/eta filled in more strict condition
    sv_is_iso_all_fsrphs = check_clean_from_obj(sv_lo, fsrph_qual_lo)

    # define isCleaned SV
    sv_is_cleaned = (ak.fill_none(sv_is_iso_all_tightleps, True)) & sv_is_iso_all_candleps & (ak.fill_none(sv_is_iso_all_fsrphs, True))
    return sv_is_cleaned