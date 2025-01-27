import os
import glob
import json
import re
import pickle
from datetime import timedelta
from copy import deepcopy as copy
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lin
import matplotlib.patches as ptc

from flame_utils import ModelFlame
from .utils import get_Dnum_from_pv, post_process_BPMdf, datetime_from_Ymd_HMS, from_listdict_to_pd#, sort_by_Dnum
from .flame_helper import convert, get_FMelem_from_PVs



def get_most_recent_matching_file(directory, file_suffix='_reconst_input_3d.json'):
    # Get all files in the directory matching the given suffix
    matching_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(file_suffix) and os.path.isfile(os.path.join(directory, f))
    ]
    
    if not matching_files:  # If no matching files are found
        return None
    
    # Find the most recent file based on modification time
    most_recent_file = max(matching_files, key=os.path.getmtime)
    return most_recent_file
    

def get_most_recent_reconst_data(ISAAC_data_rel_path,
                                 ISAAC_database_path,
                                 reconst_input_file_suffix='_reconst_input_3d.json'):
    path = os.path.join(ISAAC_database_path,ISAAC_data_rel_path)
    f_reconst_input = get_most_recent_matching_file(path,file_suffix=reconst_input_file_suffix)
    if f_reconst_input is None:
        return None
    f_summary = f_reconst_input.replace('_reconst_input_3d.json','.json')
    f_reconst_output = f_reconst_input.replace('_reconst_input_3d.json','_reconst_output_3d')
    f_fm = os.path.join(f_reconst_output,'flame_reconst_input.lat')
    f_recon_out_ = os.path.basename(f_reconst_output)
    f_reconst_output = os.path.join(f_reconst_output,f_recon_out_+'.json')
    with open(f_reconst_input, 'r') as f:
        reconst_input = json.load(f)
    with open(f_summary, 'r') as f:
        summary = json.load(f)
    with open(f_reconst_output, 'r') as f:
        reconst_output = json.load(f)
    return {'reconst_summary':summary, 
            'reconst_input'  :reconst_input, 
            'reconst_output' :reconst_output, 
            'fmlatfile'      :f_fm}

       
def get_reconst_summary_from_ISAAC_data_rel_path(
    ISAAC_data_rel_path,
    ISAAC_database_path,
    ):
    
    # find the best reconst_output folder
    ISAAC_data_path = os.path.join(ISAAC_database_path,ISAAC_data_rel_path)
    folders = [item for item in os.listdir(ISAAC_data_path) if os.path.isdir(os.path.join(ISAAC_data_path, item))]
    reconst_rel_path = None
    for folder in folders:
        if not os.path.isdir(os.path.join(ISAAC_data_path, folder)):
            continue
        if 'reconst_output' in folder:
            if reconst_rel_path is not None:
                if not( ('maruta'     in reconst_rel_path and 'maruta' in folder  and len(folder)>len(reconst_rel_path)) or \
                        ('maruta' not in reconst_rel_path and 'maruta' in folder) or \
                        ('maruta' not in reconst_rel_path and len(folder)>len(reconst_rel_path))
                      ):
                    continue
            # find the best reconst_output file
            reconst_fname = None
            for fname in os.listdir(os.path.join(ISAAC_data_path,folder)):
                if 'reconst_output' in fname and fname[:8]=='summary_' and fname[-5:]=='.json':
                    if reconst_fname is None:
                        reconst_fname = fname
                    elif len(reconst_fname) < fname:
                        reconst_fname = fname
            if reconst_fname is None:
                continue
            else:
                reconst_rel_path = folder
    if reconst_rel_path:            
        return json.load(open(os.path.join(ISAAC_data_path, reconst_rel_path, reconst_fname),'r'))
    else:
        raise ValueError(f'reconstruct summary could not found in {ISAAC_data_rel_path}')
    
    
def get_flame_evals_n_goals_from_reconst_summary(
    reconst_summary = None,
    ISAAC_data_rel_path = None,
    ISAAC_database_path = None,
    ignore_coupling = False,
    fm = None,
    return_flame=False):
    '''
    flame_evals = dict of keys: ['fm_name','fm_index','fm_elem','fm_field','fm_value','mp_field','mp_value']
    flame_goals = dict of keys: ['fm_name','fm_index','fm_elem','xrms','yrms','cxy']
    '''
    if ignore_coupling:
        _bmstateKeys = ["xrms","yrms"]
    else:
        _bmstateKeys = ["xrms","yrms","cxy"]
    if reconst_summary is None:
        assert ISAAC_data_rel_path is not None
        assert ISAAC_database_path is not None
        reconst_summary = get_reconst_summary_from_ISAAC_data_rel_path(ISAAC_data_rel_path = ISAAC_data_rel_path, 
                                                                      ISAAC_database_path = ISAAC_database_path)
    if fm is None:
        flame_lat = reconst_summary['reconst_output']['flat']
        fm_relpath = os.path.relpath(flame_lat, "/files/shared/ap/ISAAC/data")
        i_rel = fm_relpath.find('/')
        assert i_rel > 0
        if os.path.exists(flame_lat):
            fm = ModelFlame(flame_lat)
        else:
            if ISAAC_data_rel_path is not None:
                flame_lat = os.path.join("/files/shared/ap/ISAAC/data",ISAAC_data_rel_path,fm_relpath[i_rel+1:])
                if os.path.exists(flame_lat):
                    fm = ModelFlame(flame_lat)
                else:
                    flame_lat = os.path.join(ISAAC_database_path,ISAAC_data_rel_path,fm_relpath[i_rel+1:])
                    if os.path.exists(flame_lat):
                        fm = ModelFlame(flame_lat)
            else:
                flame_lat = os.path.join(ISAAC_database_path, fm_relpath)
                if os.path.exists(flame_lat):
                    fm = ModelFlame(flame_lat)
    if fm is None:
        raise ValueError(f"{flame_lat} does not exist")
    
            
    # check if envelope is reconstructed
    if reconst_summary['reconst_input']['opt_target'][-1]!='moment1':
        raise ValueError(f'reconst summary file {fname} does not conatin envelope reconsturction data')
        
    measurements = reconst_summary["meas"]    
    measurements_select = {key:np.array(reconst_summary['reconst_input']['measurement_select'][-1][key])>0
                           for key in _bmstateKeys}
    for k in _bmstateKeys:
        assert len(measurements) == len(measurements_select[k])
    
    eval_multi_cols = [(name, val_field[1]) for name, val_field in measurements[0]['flamevall'].items()]
    eval_names  = list(set(measurements[0]['flamevall'].keys()))
    eval_index  = {name:{'index':fm.get_index_by_name(name)[name][0]} for name in eval_names}
    flame_evals = {tupl:[] for tupl in eval_multi_cols}
    
#     goal_multi_cols = [(name, g) for g in goal.keys() for name, goals in measurements[0]['monitorl'].items()]
    goal_multi_cols = [(name, g) for name, goals in measurements[0]['monitorl'].items() for g in goals.keys() if g in _bmstateKeys]
    goal_names  = list(set(measurements[0]['monitorl'].keys()))
    goal_index  = {name:{'index':fm.get_index_by_name(name)[name][0]} for name in goal_names}
    flame_goals = {tupl:[] for tupl in goal_multi_cols}
    
    for i,meas in enumerate(measurements):
        if not np.any(measurements_select[k][i] for k in _bmstateKeys):
            continue
        is_filled = {col:False for col in eval_multi_cols}
        for pv, val_field in meas["flamevall"].items():
            flame_evals[(pv,val_field[1])].append(val_field[0])
            is_filled[(pv,val_field[1])] = True
        for col in eval_multi_cols:
            if not is_filled[col]:
                flame_evals[col].append(np.nan)
        is_filled = {col:False for col in goal_multi_cols}
        for mon, meas_dict in meas["monitorl"].items():
            for bmstat,v in meas_dict.items():
                if bmstat in _bmstateKeys:
                    if measurements_select[bmstat][i]:
                        flame_goals[(mon,bmstat)].append(v)
                        is_filled[(mon,bmstat)] = True
            for col in goal_multi_cols:
                if not is_filled[col]:
                    flame_goals[col].append(np.nan) 
    flame_evals = {'info': eval_index,
                   'df'  : pd.DataFrame(flame_evals)}
    flame_goals = {'info': goal_index,
                   'df'  : pd.DataFrame(flame_goals)}
    if return_flame:
        return flame_evals, flame_goals, fm
    else:
        return flame_evals, flame_goals


def get_ISAAC_BPM_data_df(path: str, 
                          from_Dnum: int = 1001, 
                          to_Dnum  : int = 9999,
                          fill_NaN_for_suspicious_beamloss_based_on_MAG: bool = False) -> Union[Dict, None]:
    """
    Extract BPM data from ISAAC files and organize it into a DataFrame.
    Args:
        path (str): Path where ISAAC BPM data files are located.
        from_Dnum (int): Starting D number for filtering.
        to_Dnum (int): Ending D number for filtering.

    Returns:
        dict or None: BPM data organized in a dictionary: Dict[str, Union[List[pd.DataFrame], pd.DataFrame]]

    This function searches for BPM data files in the specified path, loads the data,
    and organizes it into a DataFrame. The resulting dictionary contains put and get PV values,
    file name, and calculated values for BPMs.
    """
    bpm_data_files = list(glob.glob(os.path.join(path, '*bpm4pick*')))
    if len(bpm_data_files) == 0:
        print('BPM data file not found. Return None')
        return None
    else:
        filtered_bpm_data_files = []
        pattern = re.compile(r'bpm4pick_20(\d{6})_(\d{6}).')
        for f in bpm_data_files:           
            if pattern.search(f):
                filtered_bpm_data_files.append(f)
        bpm_data_files = [filtered_bpm_data_files[0]]
        BPM_rawdata = None
        for file_name in bpm_data_files:
            if file_name[-5:] == '.json':
                BPM_rawdata = json.load(open(file_name, 'r'))
            elif file_name[-4:] == '.pkl':
                BPM_rawdata = pickle.load(open(file_name, 'rb'))
            if BPM_rawdata is not None:
                if len(bpm_data_files)>1:
                    print(f'Multiple BPM data file found. Will use: {file_name}')
                break
    df = pd.DataFrame(BPM_rawdata[0]['BPMdata'])
    df = post_process_BPMdf(df,from_Dnum,to_Dnum,fill_NaN_for_large_MAG_err=True,remove_TISRAW=False,index=0)
    BPMnames = df.columns.get_level_values(0).unique().values.tolist()

    output = {'getPVvall': [],
              'putPVvall': [],
              'values': [],
              }
    Qs = []
    for i in range(len(BPM_rawdata)):
        df = pd.DataFrame(BPM_rawdata[i]['BPMdata'])
        means = post_process_BPMdf(df,from_Dnum,to_Dnum,fill_NaN_for_large_MAG_err=True,index=i)
        output['putPVvall'].append(pd.DataFrame(BPM_rawdata[i]['putPVvall'], index=[i]))

        if 'getPVvall' in BPM_rawdata[i]:
            output['getPVvall'].append(pd.DataFrame(BPM_rawdata[i]['getPVvall'], index=[i]))
        else:
            output['getPVvall'].append(pd.DataFrame(BPM_rawdata[i]['putPVvall'], index=[i]))
            output['getPVvall'][-1].columns = BPM_rawdata[-1]['getPVvall'].keys()
        output['values'].append(means)

    output['values'] = pd.concat(output['values'])
    output['putPVvall'] = pd.concat(output['putPVvall'])
    output['getPVvall'] = pd.concat(output['getPVvall'])
    output['values'] = output['values'][BPMnames]
    
    if fill_NaN_for_suspicious_beamloss_based_on_MAG:
        df = output['values']
#         pd.set_option('future.no_silent_downcasting', True) # suppress warning regarding ffill()
        for name in BPMnames:
            # Standard mean error of BPM_MAG should be less than 5%
#             too_large_mean_err = df[name]['MAG_err'] > 0.05 * df[name]['MAG']

            # Beam loss: BPM_MAG (+mean_err) should not be less than 90% of 0.9 quantile over samples
            is_beam_loss = df[name]['MAG'] +df[name]['MAG_err'] < 0.9 * df[name]['MAG'].quantile(0.9)

            # Filling NaN values to suspicious data entry
#             mask = np.logical_or(too_large_mean_err, is_beam_loss)
            mask = is_beam_loss
            df.loc[mask, name] = df[name][mask].ffill()
    output['values'] = output['values'].apply(pd.to_numeric, errors='ignore')
    return output


def filter_err_out_from_ISAAC_summary(ISAAC_summary: Dict) -> Tuple[int, Union[Dict, None]]:
    """
    This function validates the presence of required keys in the ISAAC summary data.
    It filters out scan data entries with errors and ensures the correct data types.
    
    Parameters:
    - ISAAC_summary (dict): The ISAAC summary data to be validated.

    Returns:
    tuple: A tuple containing two elements:
        - dict: The ISAAC summary data with filtered scan data.
    """
    required_keys = ['scan_type', 'monitorl', 'initputPVvall', 'initgetPVvall', 'scan_data']
    
    # Check for the presence of required keys
    if not all(key in ISAAC_summary for key in required_keys):
        return 0, None
    
    # Filter out scan_data with errors
    filtered_scan_data = []
    for data in ISAAC_summary['scan_data']:
        # Check scan data types
        if all(key in data for key in ['putPVvall', 'getPVvall', 'res_monitorl']) and \
           isinstance(data['putPVvall'], dict) and \
           isinstance(data['getPVvall'], dict) and \
           isinstance(data['res_monitorl'], list):
            if any("error" in res_monitor for res_monitor in data['res_monitorl']):
                continue
            filtered_scan_data.append(data)

    # Update ISAAC_summary_data with filtered scan_data
    ISAAC_summary_data['scan_data'] = filtered_scan_data
    return ISAAC_summary_data


def get_related_ISAAC_data_rel_path(ISAAC_data_rel_path, 
                                    ISAAC_database_path, 
                                    within_minutes = 240,
                                    filter_out_path_with_reconstruct = True):
    segment = None
    for seg in ['BDS_BTS','LS3_BTS','FS1_BMS','FS1_CSS','FE_MEBT']:
        if seg in ISAAC_data_rel_path:
            segment = seg
            break
    if segment is None:
        raise ValueError(f'segment could not identifyied from {ISAAC_data_rel_path}')
        
    ref_datetime = datetime_from_Ymd_HMS(ISAAC_data_rel_path)
    data_path_l = [path for path in os.listdir(ISAAC_database_path) 
                   if segment in path and path != ISAAC_data_rel_path]
    data_path_l = [path for path in data_path_l 
                   if abs(datetime_from_Ymd_HMS(path) -ref_datetime) < timedelta(minutes=within_minutes)]
    if filter_out_path_with_reconstruct:
        tmp = []
        for path in data_path_l:
            try:       
                reconst_summary = get_reconst_summary_from_ISAAC_reldata_path(path,ISAAC_database_path=ISAAC_database_path)
            except:    # if reconstruct summary is not found
                tmp.append(path)
        data_path_l = tmp      
    return data_path_l

