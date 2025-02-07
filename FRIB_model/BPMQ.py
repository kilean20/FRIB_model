import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .ISAAC_helper import get_ISAAC_BPM_data_df, get_flame_evals_n_goals_from_reconst_summary, get_reconst_summary_from_ISAAC_data_rel_path, get_related_ISAAC_data_rel_path
from .flame_helper import evaluate_flame_evals, from_flame_evals_to_machine_evals, fit_moment1, from_machine_evals_to_flame_evals
from .utils import get_Dnum_from_pv, get_consistent_row_from_two_BPMdf, match_rows_of_two_df, sort_by_Dnum
from collections import OrderedDict


def get_BPMQ_training_data_from_ISAAC_data(ISAAC_data_rel_path,
                                           ISAAC_database_path,
                                           from_element = None,
                                           to_element   = None,
                                           ignore_coupling = False,
                                           plot_data    = False,
                                           plot_Qlim    = None,
                                           plot_xylim   = None,
                                           return_flameinfo = False,
                                          ):
    reconst_summary = get_reconst_summary_from_ISAAC_data_rel_path(
        ISAAC_data_rel_path = ISAAC_data_rel_path, 
        ISAAC_database_path = ISAAC_database_path)
    fm_evals, fm_goals, fm = get_flame_evals_n_goals_from_reconst_summary(
        ISAAC_data_rel_path = ISAAC_data_rel_path, 
        ISAAC_database_path = ISAAC_database_path,
        reconst_summary = reconst_summary,
        ignore_coupling = ignore_coupling,
        return_flame = True)
    machine_evals = from_flame_evals_to_machine_evals(fm_evals['df'])
    
    all_indices = [info['index'] for name, info in fm_evals['info'].items()]
    all_indices +=[info['index'] for name, info in fm_goals['info'].items()] 
    i_from_element = min(all_indices)
    i_to_element   = max(all_indices)

    from_Dnum = None
    if from_element is None:
        for i in range(i_from_element,0,-1):
            try:
                from_Dnum = get_Dnum_from_pv(fm.get_element(index=i)[0]['properties']['name'])
                if from_Dnum:
                    i_from_element = i
                    break
            except:
                continue
        i_1before_from_element = None
    else:
        assert isinstance(from_element,str)
        i_from_element = fm.get_index_by_name(from_element)[from_element][0]
        i_1before_from_element = i_from_element-1
        from_Dnum = get_Dnum_from_pv(from_element)
    if from_Dnum is None:
        raise ValueError('from_Dnum could not determined automatically.')

    to_Dnum = None
    if to_element is None:
        last_elem_name = fm.get_all_names()[-1]
        i_last = fm.get_index_by_name(last_elem_name)[last_elem_name][-1]
        for i in range(i_to_element,i_last):
            try:
                to_element = fm.get_element(index=i)[0]['properties']['name']
                to_Dnum = get_Dnum_from_pv(to_element)
                if to_Dnum:
                    i_to_element = i
                    break
            except:
                continue
    else:
        assert isinstance(to_element,str)
        i_to_element = fm.get_index_by_name(to_element)[to_element][0]
        to_Dnum = get_Dnum_from_pv(to_element)
    if to_Dnum is None:
        raise ValueError('to_Dnum could not determined automatically.')
    
    if i_1before_from_element:
        _, from_bmstate = fm.run(to_element=i_1before_from_element)
    else:
        from_bmstate = fm.bmstate
    fit_result = fit_moment1(
                 fm,fm_evals,fm_goals,
                 from_bmstate = from_bmstate,
                 from_element= i_1before_from_element,
                 to_element  = i_to_element,
                 n_try = 20,
                 stop_criteria = 0.5,
                 start_fit_from_current_bmstate = True,
                 plot_fitting_quality = False,
                )
        
    BPMname_fmindex = {elem['properties']['name']:elem['index'] 
                     for elem in fm.get_element(type='bpm') 
                     if from_Dnum <= get_Dnum_from_pv(elem['properties']['name']) <= to_Dnum}

    BPMnames = list(BPMname_fmindex.keys())
    monitor_indices = list(BPMname_fmindex.values()) + [v['index'] for k,v in fm_goals['info'].items()]
    monitor_names =  BPMnames + list(fm_goals['info'].keys())

    flame_results = evaluate_flame_evals(
        fm_evals,fm,from_bmstate,
        monitor_indices = monitor_indices,
        monitor_names   = monitor_names,
        from_element    = i_1before_from_element,
        to_element      = i_to_element,
    )
    
    BPMdata = get_ISAAC_BPM_data_df(
        os.path.join(ISAAC_database_path,ISAAC_data_rel_path),
        to_Dnum   = to_Dnum,
        fill_NaN_for_suspicious_beamloss_based_on_MAG = True)
    related_rel_path = get_related_ISAAC_data_rel_path(
                            ISAAC_data_rel_path = ISAAC_data_rel_path,
                            ISAAC_database_path = ISAAC_database_path)

    df_ref = BPMdata['values'].copy()
    BPMdata['values'] = BPMdata['values'][BPMnames]

    for rel_path in related_rel_path:
        print("comparing BPM data with")
        print("    " + rel_path)
        tmp = get_ISAAC_BPM_data_df(
            os.path.join(ISAAC_database_path,rel_path),
            to_Dnum  = to_Dnum,
            fill_NaN_for_suspicious_beamloss_based_on_MAG = True)
        if tmp is None:
            continue
        try:
            consistent_rows = get_consistent_row_from_two_BPMdf(
                tmp['values'],
                df_ref,
                should_consistent_upto_Dnum = from_Dnum-1,
                fill_NaN_inconsistent_row_of_df_test=True)
        except Exception as e:
            print(e)
            continue
        if not np.any(consistent_rows):
            continue

        tmp['values'] = tmp['values'][BPMnames]    
        for k in tmp.keys():
            tmp[k] = tmp[k].iloc[1:,:][consistent_rows[1:]]
            BPMdata[k] = pd.concat((BPMdata[k],tmp[k]),ignore_index=True)
    
    BPMdfs = [BPMdata['values']]
    Reconst_dfs = [fm_goals['df']]

    df_a, df_b, list_of_df_a_like, list_of_df_b_like = match_rows_of_two_df(
        df_a = BPMdata['getPVvall'],
        df_b = machine_evals,
        cols = machine_evals.columns,
        list_of_df_a_like=BPMdfs,
        list_of_df_b_like=Reconst_dfs)
    
    measurements = pd.concat((list_of_df_a_like[0],list_of_df_b_like[0]),axis=1)
    fm_evals_new = from_machine_evals_to_flame_evals(df_a,fm)
    simulations  = evaluate_flame_evals(
        fm_evals_new,fm,from_bmstate,
        monitor_indices = monitor_indices,
        monitor_names   = monitor_names,
        from_element    = i_1before_from_element,
        to_element      = i_to_element,
    )
    
    for mon in monitor_names:
        if ':BPM_D' in mon and mon in measurements and mon in simulations:
            if set(['XPOS','YPOS','Q']).issubset(set(measurements[mon].columns)) and \
               set(['xrms','yrms']).issubset(set(simulations[mon].columns)):
                meas = measurements[mon]
                meas_beam_qMoment = (241*meas['Q'] -(meas['XPOS']**2 - meas['YPOS']**2))
                meas_beam_qMoment_err = (  (241*meas['Q_err'])**2 
                                          +(2*meas['XPOS']*meas['XPOS_err'])**2 
                                          +(2*meas['YPOS']*meas['YPOS_err'])**2 
                                        )**0.5
                sim_beam_qMoment  = simulations[mon]['xrms']**2 - simulations[mon]['yrms']**2
                measurements[(mon,'beamQ')] = meas_beam_qMoment.values
                measurements[(mon,'beamQ_err')] = meas_beam_qMoment_err.values
                simulations [(mon,'beamQ')] = sim_beam_qMoment.values
    
    meas_mons = sort_by_Dnum(list(measurements.columns.get_level_values(0).unique()))
    measurements = measurements[meas_mons]
    sim_mons = sort_by_Dnum(list(simulations.columns.get_level_values(0).unique()))
    simulations = simulations[sim_mons]
    
        
    output = {'measurement':measurements,
              'simulation' :simulations,
              'fm_evals'   :fm_evals_new['df'],
              'fit_result' :fit_result}
    
    if plot_data:
        output['figs'] = []
        for mon in monitor_names:
            if mon in measurements and mon in simulations:
                meas_set = set(measurements[mon].columns)
                sim_set  = set(simulations [mon].columns)
                meas_set.discard('beamQ')
                sim_set.discard('beamQ')
                if len(meas_set.intersection(sim_set)) > 0:
                    fig,ax = plt.subplots(figsize=(5,2.5),dpi=96)
                    for i, col in enumerate(measurements[mon]):
                        if col in simulations[mon]:
                            ax.plot(simulations [mon][col],    color='C'+str(i),label = col)
                            if col+'_err' in measurements[mon].columns:
                                ax.errorbar(x   = np.arange(len(meas_beam_qMoment)), 
                                            y   = measurements[mon][col],
                                            yerr= measurements[mon][col+'_err'],
                                            color = 'C'+str(i),
                                            fmt='.', linestyle='',
                                            capsize=5,
                                            )
                            else:
                                ax.plot(measurements[mon][col],'.',color='C'+str(i))
                    icolon = mon.find(":")
                    if icolon > 0:
                        mon_short = mon[icolon+1:]
                    ax.set_title('meas VS sim, '+mon_short)
                    if plot_xylim is not None:
                        ax.set_ylim(plot_xylim[0],plot_xylim[1])
                    ax.legend()
                    ax.set_xlabel('data no.')
                    output['figs'].append(fig)
        
        for mon in monitor_names:
            if ':BPM_D' in mon and mon in measurements and mon in simulations:
                meas_set = set(measurements[mon].columns)
                sim_set  = set(simulations [mon].columns)
                if set(['beamQ','beamQ_err']).issubset(meas_set) and 'beamQ' in sim_set:
                    fig,ax = plt.subplots(figsize=(5,2.5),dpi=96)
                    ax.plot(simulations[mon]['beamQ'],label = 'beam Q-Moment')
                    ymin,ymax = ax.get_ylim()
                    ydiff = ymax-ymin
                    ax.errorbar(x   = np.arange(len(meas_beam_qMoment)), 
                                y   = measurements[mon]['beamQ'],
                                yerr= measurements[mon]['beamQ_err'],
                                fmt='.', linestyle='',
                                capsize=5,
                                )
                    if plot_Qlim is None:
                        ax.set_ylim(ymin-0.1*ydiff, ymax+0.1*ydiff)
                    else:
                        ax.set_ylim(plot_Qlim[0],plot_Qlim[1])
                    icolon = mon.find(":")
                    if icolon > 0:
                        mon_short = mon[icolon+1:]
                    ax.set_title('meas VS sim, '+mon_short)
                    ax.set_xlabel('data no.')
                    ax.set_ylabel(r'$\sigma_x^2-\sigma_y^2 \, (mm^2)$ ')
                    ax.legend()
                    fig.tight_layout()
                    output['figs'].append(fig)
                    
    if return_flameinfo:
        flameinfo = {}
        output['flameinfo'] = flameinfo
        flameinfo['fm'] = fm
        flameinfo['from_bmstate'] = from_bmstate
        flameinfo['from_element'] = from_element
        flameinfo['i_from_element'] = i_from_element
        flameinfo['to_element'] = to_element
        flameinfo['i_to_element'] = i_to_element
        flameinfo['from_Dnum'] = from_Dnum
        flameinfo['to_Dnum'] = to_Dnum
        
    return output



_BPM_TIS161_coeffs = OrderedDict([
    ("FE_MEBT:BPM_D1056", np.array([32287, 32731, 27173, 27715])),
    ("FE_MEBT:BPM_D1072", np.array([28030, 27221, 32767, 31131])),
    ("FE_MEBT:BPM_D1094", np.array([31833, 32757, 26390, 27947])),
    ("FE_MEBT:BPM_D1111", np.array([27269, 27939, 32227, 32760])),
    ("LS1_CA01:BPM_D1129", np.array([32761, 31394, 28153, 28781])),
    ("LS1_CA01:BPM_D1144", np.array([27727, 28614, 32766, 31874])),
    ("LS1_WA01:BPM_D1155", np.array([32762, 32240, 26955, 29352])),
    ("LS1_CA02:BPM_D1163", np.array([27564, 27854, 32566, 32761])),
    ("LS1_CA02:BPM_D1177", np.array([32722, 30943, 27022, 26889])),
    ("LS1_WA02:BPM_D1188", np.array([28227, 27740, 32752, 32404])),
    ("LS1_CA03:BPM_D1196", np.array([32760, 32111, 28850, 28202])),
    ("LS1_CA03:BPM_D1211", np.array([27622, 27772, 32751, 31382])),
    ("LS1_WA03:BPM_D1222", np.array([32485, 32767, 26412, 26301])),
    ("LS1_CB01:BPM_D1231", np.array([27488, 28443, 30934, 32746])),
    ("LS1_CB01:BPM_D1251", np.array([32757, 31820, 30114, 30358])),
    ("LS1_CB01:BPM_D1271", np.array([26349, 27227, 30934, 32762])),
    ("LS1_WB01:BPM_D1286", np.array([32227, 32766, 27066, 28581])),
    ("LS1_CB02:BPM_D1295", np.array([27323, 28137, 32497, 32762])),
    ("LS1_CB02:BPM_D1315", np.array([32764, 32205, 26524, 27304])),
    ("LS1_CB02:BPM_D1335", np.array([27841, 27972, 32275, 32749])),
    ("LS1_WB02:BPM_D1350", np.array([31773, 32767, 26605, 26186])),
    ("LS1_CB03:BPM_D1359", np.array([26771, 27352, 32762, 32452])),
    ("LS1_CB03:BPM_D1379", np.array([32763, 32178, 28888, 28548])),
    ("LS1_CB03:BPM_D1399", np.array([27792, 28589, 32767, 32015])),
    ("LS1_WB03:BPM_D1413", np.array([32674, 32740, 27702, 29077])),
    ("LS1_CB04:BPM_D1423", np.array([27084, 28184, 31037, 32755])),
    ("LS1_CB04:BPM_D1442", np.array([32743, 31782, 26311, 26977])),
    ("LS1_CB04:BPM_D1462", np.array([27387, 28631, 32639, 32765])),
    ("LS1_WB04:BPM_D1477", np.array([32277, 32767, 27516, 28706])),
    ("LS1_CB05:BPM_D1486", np.array([28280, 27538, 31488, 32746])),
    ("LS1_CB05:BPM_D1506", np.array([32755, 32475, 26147, 28303])),
    ("LS1_CB05:BPM_D1526", np.array([27094, 28077, 32518, 32753])),
    ("LS1_WB05:BPM_D1541", np.array([32750, 31993, 29001, 28028])),
    ("LS1_CB06:BPM_D1550", np.array([32766, 31956, 26858, 27938])),
    ("LS1_CB06:BPM_D1570", np.array([26975, 27074, 32764, 32718])),
    ("LS1_CB06:BPM_D1590", np.array([32655, 32759, 27428, 27689])),
    ("LS1_WB06:BPM_D1604", np.array([27702, 27872, 32767, 32684])),
    ("LS1_CB07:BPM_D1614", np.array([32500, 32756, 28433, 28144])),
    ("LS1_CB07:BPM_D1634", np.array([27453, 28106, 32763, 31629])),
    ("LS1_CB07:BPM_D1654", np.array([32673, 32759, 26435, 26782])),
    ("LS1_WB07:BPM_D1668", np.array([32762, 32410, 27616, 27670])),
    ("LS1_CB08:BPM_D1677", np.array([29512, 28207, 32764, 31941])),
    ("LS1_CB08:BPM_D1697", np.array([32060, 32760, 27914, 27520])),
    ("LS1_CB08:BPM_D1717", np.array([26616, 27323, 30786, 32751])),
    ("LS1_WB08:BPM_D1732", np.array([31676, 32767, 28261, 27470])),
    ("LS1_CB09:BPM_D1741", np.array([27056, 27996, 32761, 32464])),
    ("LS1_CB09:BPM_D1761", np.array([32580, 32755, 28495, 27466])),
    ("LS1_CB09:BPM_D1781", np.array([27081, 27400, 32765, 31943])),
    ("LS1_WB09:BPM_D1796", np.array([32738, 32523, 27305, 28514])),
    ("LS1_CB10:BPM_D1805", np.array([32752, 32651, 28317, 27619])),
    ("LS1_CB10:BPM_D1825", np.array([27841, 26725, 31684, 32763])),
    ("LS1_CB10:BPM_D1845", np.array([32761, 32571, 27227, 26692])),
    ("LS1_WB10:BPM_D1859", np.array([26790, 27824, 32766, 31553])),
    ("LS1_CB11:BPM_D1869", np.array([31793, 32765, 27328, 28204])),
    ("LS1_CB11:BPM_D1889", np.array([29556, 28492, 32110, 32739])),
    ("LS1_CB11:BPM_D1909", np.array([32666, 32767, 27219, 27940])),
    ("LS1_WB11:BPM_D1923", np.array([27786, 28350, 32765, 32735])),
    ("LS1_BTS:BPM_D1967", np.array([32403, 32743, 28313, 27464])),
    ("LS1_BTS:BPM_D2027", np.array([31336, 32749, 27048, 27244])),
    ("LS1_BTS:BPM_D2054", np.array([28209, 27945, 32757, 32424])),
    ("LS1_BTS:BPM_D2116", np.array([32749, 32169, 28443, 28303])),
    ("LS1_BTS:BPM_D2130", np.array([26988, 26401, 30754, 32764])),
    ("FS1_CSS:BPM_D2212", np.array([32504, 32753, 26907, 27222])),
    ("FS1_CSS:BPM_D2223", np.array([27008, 27707, 32757, 32146])),
    ("FS1_CSS:BPM_D2248", np.array([32767, 30874, 27504, 27588])),
    ("FS1_CSS:BPM_D2278", np.array([26976, 27852, 31420, 32766])),
    ("FS1_CSS:BPM_D2313", np.array([32742, 32371, 27486, 28596])),
    ("FS1_CSS:BPM_D2369", np.array([28504, 28147, 31881, 32755])),
    ("FS1_CSS:BPM_D2383", np.array([32757, 31686, 27892, 26735])),
    ("FS1_BBS:BPM_D2421", np.array([9159, 9268, 10918, 10303])),
    ("FS1_BBS:BPM_D2466", np.array([10918, 10183, 9241, 8850])),
    ("FS1_BMS:BPM_D2502", np.array([32751, 32671, 27507, 28983])),
    ("FS1_BMS:BPM_D2537", np.array([28319, 28030, 32452, 32763])),
    ("FS1_BMS:BPM_D2587", np.array([32767, 31061, 26621, 28059])),
    ("FS1_BMS:BPM_D2600", np.array([27259, 28217, 32588, 32767])),
    ("FS1_BMS:BPM_D2665", np.array([31323, 32756, 26910, 26613])),
    ("FS1_BMS:BPM_D2690", np.array([28799, 29947, 32163, 32767])),
    ("FS1_BMS:BPM_D2702", np.array([32716, 31529, 27273, 28315])),
    ("LS2_WC01:BPM_D2742", np.array([28000, 27046, 32765, 32351])),
    ("LS2_WC02:BPM_D2782", np.array([31987, 32726, 26097, 27093])),
    ("LS2_WC03:BPM_D2821", np.array([27683, 27736, 32462, 32744])),
    ("LS2_WC04:BPM_D2861", np.array([32260, 32755, 27775, 26737])),
    ("LS2_WC05:BPM_D2901", np.array([28876, 28397, 32755, 32347])),
    ("LS2_WC06:BPM_D2941", np.array([32706, 32585, 26922, 28398])),
    ("LS2_WC07:BPM_D2981", np.array([28193, 27484, 32628, 32714])),
    ("LS2_WC08:BPM_D3020", np.array([32736, 32734, 27119, 28366])),
    ("LS2_WC09:BPM_D3060", np.array([27325, 28001, 31760, 32765])),
    ("LS2_WC10:BPM_D3100", np.array([32762, 31868, 27192, 27197])),
    ("LS2_WC11:BPM_D3140", np.array([28508, 28213, 32762, 31950])),
    ("LS2_WC12:BPM_D3180", np.array([31275, 32766, 27045, 26362])),
    ("LS2_WD01:BPM_D3242", np.array([26266, 26802, 32767, 30716])),
    ("LS2_WD02:BPM_D3304", np.array([32576, 32743, 27589, 27440])),
    ("LS2_WD03:BPM_D3366", np.array([27464, 27749, 32745, 31346])),
    ("LS2_WD04:BPM_D3428", np.array([32725, 32487, 27931, 28026])),
    ("LS2_WD05:BPM_D3490", np.array([28442, 27800, 32744, 31802])),
    ("LS2_WD06:BPM_D3552", np.array([32250, 32752, 26890, 27612])),
    ("LS2_WD07:BPM_D3614", np.array([28010, 27436, 32763, 32740])),
    ("LS2_WD08:BPM_D3676", np.array([32416, 32748, 28640, 27388])),
    ("LS2_WD09:BPM_D3738", np.array([27865, 27307, 32748, 30772])),
    ("LS2_WD10:BPM_D3800", np.array([32753, 31738, 26514, 26555])),
    ("LS2_WD11:BPM_D3862", np.array([27851, 28014, 32709, 31513])),
    ("LS2_WD12:BPM_D3924", np.array([32747, 31185, 25967, 26142])),
    ("FS2_BTS:BPM_D3943", np.array([27406, 27134, 32394, 32764])),
    ("FS2_BTS:BPM_D3958", np.array([32742, 32747, 27196, 28687])),
    ("FS2_BBS:BPM_D4019", np.array([32763, 32462, 27499, 27832])),
    ("FS2_BBS:BPM_D4054", np.array([27464, 27578, 31677, 32747])),
    ("FS2_BBS:BPM_D4087", np.array([32762, 31327, 27183, 27516])),
    ("FS2_BMS:BPM_D4142", np.array([27371, 26615, 32743, 30524])),
    ("FS2_BMS:BPM_D4164", np.array([31771, 32767, 27977, 29179])),
    ("FS2_BMS:BPM_D4177", np.array([26043, 27381, 32739, 31500])),
    ("FS2_BMS:BPM_D4216", np.array([32740, 32260, 26892, 27304])),
    ("FS2_BMS:BPM_D4283", np.array([28375, 27356, 31309, 32767])),
    ("FS2_BMS:BPM_D4326", np.array([32638, 32684, 28433, 26931])),
    ("LS3_WD01:BPM_D4389", np.array([28205, 26969, 32767, 32505])),
    ("LS3_WD02:BPM_D4451", np.array([32742, 31517, 26887, 26986])),
    ("LS3_WD03:BPM_D4513", np.array([27718, 26385, 32764, 31143])),
    ("LS3_WD04:BPM_D4575", np.array([32711, 32609, 28080, 26950])),
    ("LS3_WD05:BPM_D4637", np.array([28282, 27973, 32491, 32760])),
    ("LS3_WD06:BPM_D4699", np.array([32676, 30797, 26850, 26891])),
    ("LS3_BTS:BPM_D4753", np.array([28033, 28013, 32358, 32765])),
    ("LS3_BTS:BPM_D4769", np.array([32764, 32025, 26094, 27198])),
    ("LS3_BTS:BPM_D4843", np.array([32766, 32421, 27854, 27019])),
    ("LS3_BTS:BPM_D4886", np.array([28370, 27839, 32730, 31856])),
    ("LS3_BTS:BPM_D4968", np.array([32743, 32078, 27092, 28561])),
    ("LS3_BTS:BPM_D5010", np.array([27906, 26757, 32758, 32617])),
    ("LS3_BTS:BPM_D5092", np.array([32611, 32727, 26691, 27708])),
    ("LS3_BTS:BPM_D5134", np.array([28708, 28562, 31937, 32711])),
    ("LS3_BTS:BPM_D5216", np.array([31056, 32767, 27866, 26341])),
    ("LS3_BTS:BPM_D5259", np.array([27038, 27485, 32767, 32254])),
    ("LS3_BTS:BPM_D5340", np.array([31847, 32706, 26916, 26818])),
    ("LS3_BTS:BPM_D5381", np.array([27342, 28318, 32766, 32423])),
    ("LS3_BTS:BPM_D5430", np.array([32734, 32240, 28146, 26966])),
    ("LS3_BTS:BPM_D5445", np.array([27052, 26354, 30865, 32756])),
    ("BDS_BTS:BPM_D5499", np.array([32751, 32087, 26576, 26592])),
    ("BDS_BTS:BPM_D5513", np.array([28344, 28530, 32626, 32765])),
    ("BDS_BTS:BPM_D5565", np.array([32256, 32737, 28547, 27498])),
    ("BDS_BBS:BPM_D5625", np.array([27742, 27831, 32667, 32435])),
    ("BDS_BBS:BPM_D5653", np.array([32735, 31587, 28817, 28221])),
    ("BDS_BBS:BPM_D5680", np.array([30691, 32729, 27155, 27157])),
    ("BDS_FFS:BPM_D5742", np.array([26544, 26681, 31966, 32767])),
    ("BDS_FFS:BPM_D5772", np.array([32740, 32436, 25151, 26329])),
    ("BDS_FFS:BPM_D5790", np.array([28058, 27615, 32697, 32764])),
    ("BDS_FFS:BPM_D5803", np.array([30801, 32767, 26359, 26019])),
    ("BDS_FFS:BPM_D5818", np.array([27247, 26734, 32767, 31213])),
])