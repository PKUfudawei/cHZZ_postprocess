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


@nb.njit
def pos_to_bool(target, pos, builder):
    for t, p in zip(target, pos):
        builder.begin_list()
        t_len = len(t)
        p_len = len(p)
        p_sort = sorted(p)
        p_index = 0 if p_len > 0 else -1
        for i in range(t_len):
            if p_index >= 0 and p_index < p_len and i == p_sort[p_index]:
                p_index += 1
                builder.boolean(True)
            else:
                builder.boolean(False)
        builder.end_list()
    return builder


def ntupler(df, is_hc=False, store_gen=True, store_pfcands=False, store_flat_only=False):
    if store_gen:
        _pt = ak.singletons(df.GENH_pt[:, -1])
        genh = ak.zip(dict(
            pt=_pt,
            pt_reg=np.log(_pt + 1),
            eta=ak.singletons(df.GENH_eta[:, -1]),
            phi=ak.singletons(df.GENH_phi[:, -1]),
        ))

    # GEN leptons
    nevent = len(df)
    if store_gen and (not store_pfcands):
        pass_fiducial_gen = (df.GENlep_Hindex[:,2] != -1) & (df.GENlep_Hindex[:,2] != 99)

        _genlep_unselected = ak.zip(dict(
            pt=df.GENlep_pt,
            pt_reg=np.log(df.GENlep_pt + 1),
            eta=df.GENlep_eta,
            phi=df.GENlep_phi,
            mass=df.GENlep_mass,
            mass_reg=np.log(df.GENlep_mass + 1),
            isEle=abs(df.GENlep_id) == 11,
            isMuon=abs(df.GENlep_id) == 13,
            isTau=abs(df.GENlep_id) == 15,
            isPosCharged=df.GENlep_id > 0,
            ),
        )
        _genlep = {}
        for b in ak.fields(_genlep_unselected):
            _genlep[b] = inner_dim_indexing(ak.values_astype(_genlep_unselected[b], np.float32), df.GENlep_Hindex, ak.ArrayBuilder()).snapshot()
        _genlep = ak.zip(_genlep)

        _genlep_lo = ak.zip(
            dict(
                pt=_genlep.pt,
                eta=_genlep.eta,
                phi=_genlep.phi,
                mass=_genlep.mass,
            ),
            behavior=vector.behavior,
            with_name='PtEtaPhiMLorentzVector',
        )

        _genlep_lo_mask = ak.mask(_genlep_lo, pass_fiducial_gen)
        _gen_Z1_lo = ak.singletons(_genlep_lo_mask[:, 0] + _genlep_lo_mask[:, 1])
        _gen_Z2_lo = ak.singletons(_genlep_lo_mask[:, 2] + _genlep_lo_mask[:, 3])
        _gen_H_lo = _gen_Z1_lo + _gen_Z2_lo
        gen_Z1 = ak.zip(dict(
            pt=_gen_Z1_lo.pt,
            pt_reg=np.log(_gen_Z1_lo.pt + 1),
            eta=_gen_Z1_lo.eta,
            phi=_gen_Z1_lo.phi,
            px=_gen_Z1_lo.x,
            py=_gen_Z1_lo.y,
            pz=_gen_Z1_lo.z,
            e=_gen_Z1_lo.t,
            mass=_gen_Z1_lo.mass,
            mass_reg=np.log(_gen_Z1_lo.mass + 1),
            ptrel=_gen_Z1_lo.pt / _gen_H_lo.pt,
            deta=_gen_Z1_lo.eta - _gen_H_lo.eta,
            dphi=_gen_Z1_lo.phi - _gen_H_lo.phi,
        ))
        gen_Z2 = ak.zip(dict(
            pt=_gen_Z2_lo.pt,
            pt_reg=np.log(_gen_Z2_lo.pt + 1),
            eta=_gen_Z2_lo.eta,
            phi=_gen_Z2_lo.phi,
            px=_gen_Z2_lo.x,
            py=_gen_Z2_lo.y,
            pz=_gen_Z2_lo.z,
            e=_gen_Z2_lo.t,
            mass=_gen_Z2_lo.mass,
            mass_reg=np.log(_gen_Z2_lo.mass + 1),
            ptrel=_gen_Z2_lo.pt / _gen_H_lo.pt,
            deta=_gen_Z2_lo.eta - _gen_H_lo.eta,
            dphi=_gen_Z2_lo.phi - _gen_H_lo.phi,
        ))
        genlep = _genlep
        _gen_H_lo_4 = ak.concatenate([_gen_H_lo for _ in range(4)], axis=1)
        genlep['px'] = _genlep_lo.x
        genlep['py'] = _genlep_lo.y
        genlep['pz'] = _genlep_lo.z
        genlep['e'] = _genlep_lo.t
        genlep['ptrel'] = genlep.pt / _gen_H_lo_4.pt
        genlep['deta'] = genlep.eta - _gen_H_lo_4.pt
        genlep['dphi'] = genlep.phi - _gen_H_lo_4.phi


    # GEN B/C Hadrons
    if store_gen:
        _absid = abs(df.GENPart_id)
        _genparticles = ak.zip(dict(
            pt=df.GENPart_pt,
            pt_reg=np.log(df.GENPart_pt + 1),
            eta=df.GENPart_eta,
            phi=df.GENPart_phi,
            id=df.GENPart_id,
            isb=((_absid==5) | ((_absid>=50) & (_absid<60)) | ((_absid>=500) & (_absid<600)) | ((_absid>=5000) & (_absid<6000))),
            isc=((_absid==4) | ((_absid>=40) & (_absid<50)) | ((_absid>=400) & (_absid<500)) | ((_absid>=4000) & (_absid<5000))),
            isc_from_b=df.GENPart_isCFromBHad,
            is_excite=df.GENPart_isExcite,
            # use truth higgs pt...
            ptrel=df.GENPart_pt / genh.pt[:, 0],
            deta=df.GENPart_eta - genh.eta[:, 0],
            dphi=df.GENPart_phi - genh.phi[:, 0],
        ))
        _genpart_cut = (_genparticles.is_excite == False)
        genpartons = _genparticles[_genpart_cut & (_absid < 10)]
        genhadrons = _genparticles[_genpart_cut & (_absid > 10)]

    ###########
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

    _lep_lo_mask = ak.mask(_lep_lo, pass_fiducial)
    _Z1_lo = ak.singletons(_lep_lo_mask[:, 0] + _lep_lo_mask[:, 1])
    _Z2_lo = ak.singletons(_lep_lo_mask[:, 2] + _lep_lo_mask[:, 3])
    _H_lo = _Z1_lo + _Z2_lo
    Z1 = ak.zip(dict(
        pt=_Z1_lo.pt,
        pt_reg=np.log(_Z1_lo.pt + 1),
        eta=_Z1_lo.eta,
        phi=_Z1_lo.phi,
        px=_Z1_lo.x,
        py=_Z1_lo.y,
        pz=_Z1_lo.z,
        e=_Z1_lo.t,
        mass=_Z1_lo.mass,
        mass_reg=np.log(_Z1_lo.mass + 1),
        ptrel=_Z1_lo.pt / _H_lo.pt,
        deta=_Z1_lo.eta - _H_lo.eta,
        dphi=_Z1_lo.phi - _H_lo.phi,
    ))
    Z2 = ak.zip(dict(
        pt=_Z2_lo.pt,
        pt_reg=np.log(_Z2_lo.pt + 1),
        eta=_Z2_lo.eta,
        phi=_Z2_lo.phi,
        px=_Z2_lo.x,
        py=_Z2_lo.y,
        pz=_Z2_lo.z,
        e=_Z2_lo.t,
        mass=_Z2_lo.mass,
        mass_reg=np.log(_Z2_lo.mass + 1),
        ptrel=_Z1_lo.pt / _H_lo.pt,
        deta=_Z1_lo.eta - _H_lo.eta,
        dphi=_Z1_lo.phi - _H_lo.phi,
    ))
    H = ak.zip(dict(
        pt=_H_lo.pt,
        pt_reg=np.log(_H_lo.pt + 1),
        eta=_H_lo.eta,
        phi=_H_lo.phi,
        px=_H_lo.x,
        py=_H_lo.y,
        pz=_H_lo.z,
        e=_H_lo.t,
        mass=_H_lo.mass,
        mass_reg=np.log(_H_lo.mass + 1),
    ))
    lep = _lep
    _H_lo_4 = ak.concatenate([_H_lo for _ in range(4)], axis=1)
    lep['ptrel'] = lep.pt / _H_lo_4.pt
    lep['deta'] = lep.eta - _H_lo_4.pt
    lep['dphi'] = lep.phi - _H_lo_4.phi

    ## Jets
    _H_lo_mask = _H_lo.mask[ak.num(_H_lo) > 0]
    _H_pt = ak.fill_none(_H_lo_mask.pt[:, 0], -1)
    _H_eta = ak.fill_none(_H_lo_mask.eta[:, 0], 0)
    _H_phi = ak.fill_none(_H_lo_mask.phi[:, 0], 0)
    jet = ak.zip(dict(
        pt=df.jet_pt,
        pt_reg=np.log(df.jet_pt + 1),
        eta=df.jet_eta,
        phi=df.jet_phi,
        mass=df.jet_mass,
        mass_reg=np.log(df.jet_mass + 1),
        ptrel=df.jet_pt / _H_pt,
        deta=df.jet_eta - _H_eta,
        dphi=df.jet_phi - _H_phi,
        ParticleNet_b=df.jet_ParticleNet_b,
        ParticleNet_bb=df.jet_ParticleNet_bb,
        ParticleNet_c=df.jet_ParticleNet_c,
        ParticleNet_cc=df.jet_ParticleNet_cc,
        ParticleNet_uds=df.jet_ParticleNet_uds,
        ParticleNet_g=df.jet_ParticleNet_g,
        ParticleNet_undef=df.jet_ParticleNet_undef,
        ParticleNet_pu=df.jet_ParticleNet_pu,
        ParticleNet_CvsL=df.jet_ParticleNet_CvsL,
        ParticleNet_CvsB=df.jet_ParticleNet_CvsB,
        ),
    )
    jet_lo = ak.zip(
        dict(
            pt=jet.pt,
            eta=jet.eta,
            phi=jet.phi,
            mass=jet.mass,
        ),
        behavior=vector.behavior,
        with_name='PtEtaPhiMLorentzVector',
    )
    jet['px'] = jet_lo.x
    jet['py'] = jet_lo.y
    jet['pz'] = jet_lo.z
    jet['e'] = jet_lo.t
    cleanedjet = jet[df.jet_iscleanH4l]
    ## Starting from v3: add pT > 15 cut
    cleanedjet = cleanedjet[cleanedjet.pt > 15]
    # jet['isCleaned'] = pos_to_bool(df.jet_pt, df.jet_iscleanH4l, ak.ArrayBuilder()).snapshot()

    ## SVs
    sv = ak.zip(dict(
        pt=df.sv_pt,
        pt_reg=np.log(df.sv_pt + 1),
        eta=df.sv_eta,
        phi=df.sv_phi,
        mass=df.sv_mass,
        mass_reg=np.log(df.sv_mass + 1),
        ptrel=df.sv_pt / _H_pt,
        deta=df.sv_eta - _H_eta,
        dphi=df.sv_phi - _H_phi,
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
    cleanedsv = sv[sv_is_cleaned]
    sv['isCleaned'] = sv_is_cleaned

    if store_pfcands:
        pfcand = ak.zip(dict(
            puppiw=df.pfcand_puppiw,
            pt=df.pfcand_pt,
            e=df.pfcand_e,
            eta=df.pfcand_eta,
            phi=df.pfcand_phi,
            pt_reg=np.log(df.pfcand_pt + 1),
            e_reg=np.log(df.pfcand_e + 1),
            ptrel=df.pfcand_pt / _H_pt,
            deta=df.pfcand_eta - _H_eta,
            dphi=df.pfcand_phi - _H_phi,

            hcalFrac=df.pfcand_hcalFrac,
            VTX_ass=df.pfcand_VTX_ass,
            lostInnerHits=df.pfcand_lostInnerHits,
            quality=df.pfcand_quality,
            charge=df.pfcand_charge,
            isEl=df.pfcand_isEl,
            isMu=df.pfcand_isMu,
            isChargedHad=df.pfcand_isChargedHad,
            isGamma=df.pfcand_isGamma,
            isNeutralHad=df.pfcand_isNeutralHad,
            drminsv=df.pfcand_drminsv,
            normchi2=df.pfcand_normchi2,
            dz=df.pfcand_dz,
            dzsig=df.pfcand_dzsig,
            dxy=df.pfcand_dxy,
            dxysig=df.pfcand_dxysig,
            dptdpt=df.pfcand_dptdpt,
            detadeta=df.pfcand_detadeta,
            dphidphi=df.pfcand_dphidphi,
            dxydxy=df.pfcand_dxydxy,
            dzdz=df.pfcand_dzdz,
            dxydz=df.pfcand_dxydz,
            dphidxy=df.pfcand_dphidxy,
            dlambdadz=df.pfcand_dlambdadz,
            )
        )
        pfcand_lo = ak.zip(
            dict(
                pt=pfcand.pt,
                eta=pfcand.eta,
                phi=pfcand.phi,
                e=pfcand.e,
            ),
            behavior=vector.behavior,
            with_name='PtEtaPhiELorentzVector',
        )
        pfcand['px'] = pfcand_lo.x
        pfcand['py'] = pfcand_lo.y
        pfcand['pz'] = pfcand_lo.z
        pfcand = pfcand[df.pfcand_pt > 0] # remove invalid candidates

    ###############
    # write output
    out_tree = {}
    out_tree.update({
        'Run': df.Run,
        'Event': df.Event,
        'LumiSect': df.LumiSect,
        'genWeight': df.genWeight,
        'mass4l': df.mass4l,
        'D_bkg_kin': df.D_bkg_kin,
        'is_hc': ak.zeros_like(df.Run, dtype=bool) + is_hc,
        'pass_fiducial': pass_fiducial,
    })
    if store_gen and (not store_pfcands):
        out_tree['pass_fiducial_gen'] = pass_fiducial_gen

    if not store_flat_only:
        if store_gen and (not store_pfcands):
            out_tree['GENH'] = genh
            out_tree['GENZ1'] = gen_Z1
            out_tree['GENZ2'] = gen_Z2
            out_tree['GENlep'] = genlep
        if store_gen:
            out_tree['GENparton'] = genpartons
            out_tree['GENhadron'] = genhadrons
        out_tree['H'] = H
        out_tree['Z1'] = Z1
        out_tree['Z2'] = Z2
        out_tree['lep'] = lep
        out_tree['jet'] = jet
        out_tree['cleanedjet'] = cleanedjet
        out_tree['sv'] = sv
        out_tree['cleanedsv'] = cleanedsv
        if store_pfcands:
            out_tree['pfcand'] = pfcand

    else:
        H_mask = H.mask[ak.num(H) > 0]
        Z1_mask = Z1.mask[ak.num(Z1) > 0]
        Z2_mask = Z2.mask[ak.num(Z2) > 0]
        lep_mask = lep.mask[ak.num(lep) == 4]

        for b in ak.fields(H):
            out_tree[f'H_{b}'] = ak.fill_none(H_mask[b][:, 0], -1)
        for b in ak.fields(Z1):
            out_tree[f'Z1_{b}'] = ak.fill_none(Z1_mask[b][:, 0], -1)
        for b in ak.fields(Z2):
            out_tree[f'Z2_{b}'] = ak.fill_none(Z2_mask[b][:, 0], -1)
        for b in ak.fields(lep):
            for i in range(4):
                out_tree[f'lep{i}_{b}'] = ak.fill_none(lep_mask[b][:, i], -1)

        out_tree['n_cleanedjet'] = ak.num(cleanedjet)
        cleanedjet_inds = ak.singletons(ak.argmax(cleanedjet.ParticleNet_c + cleanedjet.ParticleNet_cc, axis=-1))
        cleanedjet_leadc = cleanedjet.mask[ak.num(cleanedjet) > 0][cleanedjet_inds][:, 0]
        for b in ak.fields(cleanedjet):
            out_tree[f'cleanedjet_leadc2c_{b}'] = ak.fill_none(cleanedjet_leadc[b], -1)

        out_tree['n_cleanedsv'] = ak.num(cleanedsv)
        cleanedsv_inds = ak.singletons(ak.argmax(cleanedsv.ParticleNet_c + cleanedsv.ParticleNet_cc, axis=-1))
        cleanedsv_leadc = cleanedsv.mask[ak.num(cleanedsv) > 0][cleanedsv_inds][:, 0]
        for b in ak.fields(cleanedsv):
            out_tree[f'cleanedsv_leadc2c_{b}'] = ak.fill_none(cleanedsv_leadc[b], -1)

        # jet/sv + H/Z features (features added in v3)
        cleaned_jet_lo = jet_lo[df.jet_iscleanH4l]
        cleaned_jet_lo = cleaned_jet_lo[cleaned_jet_lo.pt > 15]
        cleanedjet_leadc_lo = cleaned_jet_lo.mask[ak.num(cleanedjet) > 0][cleanedjet_inds][:, 0]
        cleanedsv_lo = sv_lo[sv_is_cleaned]
        cleanedsv_leadc_lo = cleanedsv_lo.mask[ak.num(cleanedsv) > 0][cleanedsv_inds][:, 0]

        for name, obj in zip(['H', 'Z1', 'Z2'], [_H_lo.mask[ak.num(H) > 0][:, 0], _Z1_lo.mask[ak.num(Z1) > 0][:, 0], _Z2_lo.mask[ak.num(Z2) > 0][:, 0]]):
            out_tree[f'cleanedjet_leadc2c_{name}_mass'] = ak.fill_none((cleanedjet_leadc_lo + obj).mass, -99.)
            out_tree[f'cleanedjet_leadc2c_{name}_ptrel'] = ak.fill_none(cleanedjet_leadc_lo.pt / obj.pt, -99.)
            out_tree[f'cleanedjet_leadc2c_{name}_dr'] = ak.fill_none(cleanedjet_leadc_lo.delta_r(obj), -99.)
            out_tree[f'cleanedjet_leadc2c_{name}_deta'] = ak.fill_none(cleanedjet_leadc_lo.eta - obj.eta, -99.)
            out_tree[f'cleanedjet_leadc2c_{name}_dphi'] = ak.fill_none((cleanedjet_leadc_lo.phi - obj.phi + np.pi) % (2 * np.pi) - np.pi, -99.)

            out_tree[f'cleanedsv_leadc2c_{name}_mass'] = ak.fill_none((cleanedsv_leadc_lo + obj).mass, -99.)
            out_tree[f'cleanedsv_leadc2c_{name}_ptrel'] = ak.fill_none(cleanedsv_leadc_lo.pt / obj.pt, -99.)
            out_tree[f'cleanedsv_leadc2c_{name}_dr'] = ak.fill_none(cleanedsv_leadc_lo.delta_r(obj), -99.)
            out_tree[f'cleanedsv_leadc2c_{name}_deta'] = ak.fill_none(cleanedsv_leadc_lo.eta - obj.eta, -99.)
            out_tree[f'cleanedsv_leadc2c_{name}_dphi'] = ak.fill_none((cleanedsv_leadc_lo.phi - obj.phi + np.pi) % (2 * np.pi) - np.pi, -99.)

    return out_tree
