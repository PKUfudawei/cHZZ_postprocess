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


def recon_leptons(df, dfext):

    # print('++ reconstructing leptons')
    pass_fiducial = (df.lep_Hindex[:,2] != -1) & (df.lep_Hindex[:,2] != 99) & (ak.num(df.lep_pt) == ak.num(df.lep_id)) # not sure why the length is different..

    _lep_unselected = ak.zip(dict(
        pt=df.lep_pt.mask[pass_fiducial],
        eta=df.lep_eta.mask[pass_fiducial],
        phi=df.lep_phi.mask[pass_fiducial],
        mass=df.lep_mass.mask[pass_fiducial],
        id=df.lep_id.mask[pass_fiducial],
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
            id=_lep.id,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    dfext['lep_pass_fiducial'] = pass_fiducial
    dfext['lep_unselected'] = _lep_unselected
    dfext['lep'] = _lep_lo
    dfext['channel'] = ak.values_astype(
        ((abs(_lep_lo.id[:, 0])==11) & (abs(_lep_lo.id[:, 1])==11) & (abs(_lep_lo.id[:, 2])==11) & (abs(_lep_lo.id[:, 3])==11)) * 1111 + \
        ((abs(_lep_lo.id[:, 0])==13) & (abs(_lep_lo.id[:, 1])==13) & (abs(_lep_lo.id[:, 2])==13) & (abs(_lep_lo.id[:, 3])==13)) * 1313 + \
        ((abs(_lep_lo.id[:, 0])==11) & (abs(_lep_lo.id[:, 1])==11) & (abs(_lep_lo.id[:, 2])==13) & (abs(_lep_lo.id[:, 3])==13)) * 1113 + \
        ((abs(_lep_lo.id[:, 0])==13) & (abs(_lep_lo.id[:, 1])==13) & (abs(_lep_lo.id[:, 2])==11) & (abs(_lep_lo.id[:, 3])==11)) * 1311, np.int32)

def recon_cleaned_sv(df, dfext, force_all=False):

    if force_all or 'lep' not in dfext:
        recon_leptons(df, dfext)

    # print('++ setting up SVs')
    sv = ak.zip(
        dict(
            pt=df.sv_pt,
            eta=df.sv_eta,
            phi=df.sv_phi,
            mass=df.sv_mass,
            costhetasvpv=df.sv_costhetasvpv,
            ParticleNet_b=df.sv_ParticleNet_b,
            ParticleNet_bb=df.sv_ParticleNet_bb,
            ParticleNet_c=df.sv_ParticleNet_c,
            ParticleNet_cc=df.sv_ParticleNet_cc,
            ParticleNet_unmat=df.sv_ParticleNet_unmat,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    sv['px'] = sv.x
    sv['py'] = sv.y
    sv['pz'] = sv.z
    sv['e'] = sv.t
    pproj = sv.rho * np.sqrt(1 - sv.costhetasvpv * sv.costhetasvpv)
    sv['masscor'] = np.sqrt(sv.mass * sv.mass + pproj * pproj) + pproj

    def check_clean_from_obj(target, cleanded_obj):
        comb = ak.cartesian([target, cleanded_obj], nested=True)
        dr = comb['0'].delta_r(comb['1'])
        return ak.all(dr > 0.4, axis=-1)

    # clean from all tight leptons
    isoCutMu, isoCutEl = 0.35, 9999.
    passed_idiso = ((abs(df.lep_id)==13) & (df.lep_RelIso<=isoCutMu)) | ((abs(df.lep_id)==11) & (df.lep_RelIso<=isoCutEl))
    _lep_unselected_passed_idiso = dfext['lep_unselected'][passed_idiso]
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
    sv_is_iso_all_tightleps = check_clean_from_obj(sv, tightleps_unselected_lo)

    # clean from all higgs cands leptons
    sv_is_iso_all_candleps = check_clean_from_obj(sv, dfext['lep'])

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
    fsrph_qual_lo = fsrph_lo.mask[dfext['lep_pass_fiducial']][fsrph_is_qual] # should pass fiducial because fsrPhotons_pt/eta filled in more strict condition
    sv_is_iso_all_fsrphs = check_clean_from_obj(sv, fsrph_qual_lo)

    # define isCleaned SV
    sv_is_cleaned = (ak.fill_none(sv_is_iso_all_tightleps, True)) & sv_is_iso_all_candleps & (ak.fill_none(sv_is_iso_all_fsrphs, True))

    dfext['cleaned_sv'] = sv[sv_is_cleaned]


def recon_cleaned_jet(df, dfext):

    # print('++ setting up jets')
    cleaned_jet = ak.zip(
        dict(
            pt=df.jet_pt[df.jet_iscleanH4l],
            eta=df.jet_eta[df.jet_iscleanH4l],
            phi=df.jet_phi[df.jet_iscleanH4l],
            mass=df.jet_mass[df.jet_iscleanH4l],
            DeepJet_CvsL=df.jet_DeepJet_CvsL[df.jet_iscleanH4l],
            DeepJet_CvsB=df.jet_DeepJet_CvsB[df.jet_iscleanH4l],
            ParticleNet_CvsL=df.jet_ParticleNet_CvsL[df.jet_iscleanH4l],
            ParticleNet_CvsB=df.jet_ParticleNet_CvsB[df.jet_iscleanH4l],
            ParticleNet_c=df.jet_ParticleNet_c[df.jet_iscleanH4l],
            ParticleNet_cc=df.jet_ParticleNet_cc[df.jet_iscleanH4l],
            hadronFlavour=df.jet_hadronFlavour[df.jet_iscleanH4l],
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    # add pT > 15 threshold
    cleaned_jet = cleaned_jet[cleaned_jet.pt >= 15]
    dfext['cleaned_jet'] = cleaned_jet

def recon_jet_sv_candidate(df, dfext):

    if 'cleaned_jet' not in dfext:
        recon_cleaned_jet(df, dfext)
    if 'cleaned_sv' not in dfext:
        recon_cleaned_sv(df, dfext)
    
    # print('++ setting up jet & SV candidates')

    # jet candidate ordered by ParticleNet c+cc score
    jet_cand_idx = ak.argmax(dfext.cleaned_jet.ParticleNet_c + dfext.cleaned_jet.ParticleNet_cc, axis=1, keepdims=True)
    dfext['jet_cand'] = ak.firsts(dfext.cleaned_jet[jet_cand_idx])

    # SV candidate by closest
    dr = dfext.cleaned_sv.delta_r(dfext.jet_cand)
    dr = dr.mask[dr < 0.4]
    sv_cand_idx = ak.argmin(dr, axis=1, keepdims=True)
    dfext['sv_cand_closest'] = ak.firsts(dfext.cleaned_sv[sv_cand_idx])

    # and also find SV candidate by ParticleNet c+cc score
    sv_cand_idx = ak.argmax(dfext.cleaned_sv.ParticleNet_c + dfext.cleaned_sv.ParticleNet_cc, axis=1, keepdims=True)
    dfext['sv_cand'] = ak.firsts(dfext.cleaned_sv[sv_cand_idx])
