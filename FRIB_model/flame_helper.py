from copy import deepcopy as copy
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import OrderedDict

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lin
import matplotlib.patches as ptc
from IPython.display import display

from flame_utils import ModelFlame

from .machine_portal_helper import  get_MPelem_from_PVs
from .utils import NelderMead, is_list_of_lists, get_Dnum_from_pv, split_name_field_from_PV, sort_by_Dnum, _name_conversions, warn

_PV2field = {
    ':PSC1_': ('I', 'tm_ykick'),
    ':PSC2_': ('I', 'tm_xkick'),
    ':PSQ_':  ('I', 'B2')
     }

_PV2parity = {
    ':PSC1_':  1,
    ':PSC2_': -1,
    ':PSQ_':   1,
     }


def get_PVname_from_FLAMEname(fm_elem_name, dev_field=None, fm_field=None, tag='RD'):
    PV_name = None
    for dev_name, phys_name in _name_conversions:
        if phys_name in fm_elem_name:
            PV_name = fm_elem_name.replace(phys_name,dev_name) 
            field_conversion_info = _PV2field[dev_name]
    if PV_name is None:
        raise ValueError(f"PV_name could not indentified from FLAME name: {fm_elem_name}")
    
    if dev_field is None:
        dev_field = field_conversion_info[0]
        if fm_field is not None:
            if fm_field != field_conversion_info[1]:
                raise ValueError(f"PV could not indentified from FLAME name: {fm_elem_name} with fm_field: {fm_field}")
                
    return PV_name +':' + dev_field + '_' + tag


class construct_converter:
    def __init__(self):
        self.cache = {}
        self._PV2field = _PV2field
        self._PV2parity = _PV2parity
    def _get_conversion_info_from_PV(self,PVname):
        # info needed for conversion
        name, _ = split_name_field_from_PV(PVname)
        if name in self.cache:
            polarity, mp_elem, mp_field, fm_field = self.cache[name] 
        else:
            field = None
            for key, val in self._PV2field.items():
                if key in name:
                    if field is not None:
                        raise ValueError(f'multiple conversion field found for {name}')
                    field = val
            if field is None:
                raise ValueError(f'conversion field for {name} could not be automatically determined')
            polarity = None
            for key, val in self._PV2parity.items():
                if key in name:
                    if polarity is not None:
                        raise ValueError(f'multiple conversion polarity found for {name}')
                    polarity = val
            if polarity is None:
                raise ValueError(f'conversion polarity for {name} could not be automatically determined')
                
            mp_elem = get_MPelem_from_PVs([name])[0]
            mp_field = field[0]
            fm_field = field[1]
            self.cache[name] = [polarity, mp_elem, mp_field, fm_field]
        
        return polarity, mp_elem, mp_field, fm_field
            
    def fm2mp(self,
              fm_value,
              PVname,
              polarity = None,
              fm_field = None,
              mp_field = None,
              return_all_info = False,
             ):
        polarity_, mp_elem, mp_field_, fm_field_ = self._get_conversion_info_from_PV(PVname)
        if mp_field is not None:
            if not mp_field_ == mp_field:
                raise ValueError(f'machine field from PV:{PVname} is {mp_field} but field from conversion pre-set {mp_field_}. FLAME field {fm_field_} may not right')
        if fm_field is not None:
            if not fm_field_ == fm_field:
                raise ValueError(f'machine field from PV {PVname} is {mp_field} but field from conversion pre-set {mp_field_}. FLAME field {fm_field_} may not right')
        if polarity is not None:
            if not polarity_ == polarity:
                raise ValueError(f'polarity from PV {PVname} is {polarity_} but polarity from conversion pre-set {polarity}.')
        
        mp_field = mp_field or mp_field_
        fm_field = fm_field or fm_field_
        polarity = polarity or polarity_
        
        if fm_field == 'tm_xkick' or fm_field == 'tm_ykick':
            fm_field_ = 'TM'
        else:
            fm_field_ = fm_field

        mp_value = polarity*mp_elem.convert(fm_value, from_field=fm_field_, to_field=mp_field)
        if return_all_info:
            return {'polarity':polarity,
                    'mp_field':mp_field,
                    'mp_value':mp_value,
                    'fm_field':fm_field,
                    'fm_value':fm_value,
                   }
        else:
            return mp_value
    
    def mp2fm(self,
              mp_value,
              PVname,
              polarity = None,
              fm_field = None,
              mp_field = None,
              return_all_info = False,
             ):
        polarity_, mp_elem, mp_field_, fm_field_ = self._get_conversion_info_from_PV(PVname)
        if mp_field is not None:
            if not mp_field_ == mp_field:
                raise ValueError(f'machine field from PV {PVname} is {mp_field} but field from conversion pre-set {mp_field_}. FLAME field {fm_field_} may not right')
        if fm_field is not None:
            if not fm_field_ == fm_field:
                raise ValueError(f'machine field from PV {PVname} is {mp_field} but field from conversion pre-set {mp_field_}. FLAME field {fm_field_} may not right')
        if polarity is not None:
            if not polarity_ == polarity:
                raise ValueError(f'polarity from PV {PVname} is {polarity_} but polarity from conversion pre-set {polarity}.')
                
        mp_field = mp_field or mp_field_
        fm_field = fm_field or fm_field_
        polarity = polarity or polarity_
        
        if fm_field == 'tm_xkick' or fm_field == 'tm_ykick':
            fm_field_ = 'TM'
        else:
            fm_field_ = fm_field
        
#         print("polarity, mp_value, mp_field, fm_field_",polarity, mp_value, mp_field, fm_field_)
#         print("mp_elem",mp_elem)
#         print("mp_elem.convert(mp_value, from_field=mp_field, to_field=fm_field_)",mp_elem.convert(mp_value, from_field=mp_field, to_field=fm_field_))
        
        fm_value = polarity*mp_elem.convert(mp_value, from_field=mp_field, to_field=fm_field_)
        if return_all_info:
            return {'polarity':polarity,
                    'mp_field':mp_field,
                    'mp_value':mp_value,
                    'fm_field':fm_field,
                    'fm_value':fm_value,
                   }
        else:
            return fm_value


convert = construct_converter()
    

def get_FMelem_from_PVs(PVs: list[str], 
                        fm: ModelFlame) -> list:
    """
    Retrieves FLAME elements from a list of PVs.

    Args:
        PVs (list): List of PV strings.
        fm: FLAME instance.

    Returns:
        list or None: List of FLAME elements corresponding to the PVs.
    """
    names = [split_name_field_from_PV(PV, return_device_name=False)[0] for PV in PVs]

    fm_names = fm.get_all_names()
    fm_dnums = [get_Dnum_from_pv(fm_name) for fm_name in fm_names]
    elems = []
    for name in names:
        if name in fm_names:
            elem = fm.get_element(name=name)
        else:
            elem = []
        if len(elem) == 0:
            # try replaces
            for orig, new in _name_conversions:
                name_ = name.replace(orig, new)
                if name_ in fm_names:
                    elem = fm.get_element(name=name_)
                else:
                    elem = []
                if len(elem) > 0:
                    break
            # if still not found, get elem from matching dnum
            if len(elem) == 0:
                i = fm_dnums.index(get_Dnum_from_pv(name))
                if i > 0:
                    elem = fm.get_element(name=fm_names[i])
                    print(f"FLAME element finding from name {name} was not successful. The FLAME element found based on D-number is: {elem[0]['properties']['name']}")
        if len(elem) == 0:
            elems.append(None)
            print(f"FLAME element is not found for PV: {name}. 'None' padding is added in the output list")
        else:
            elems.append(elem[0])
    return elems



def zero_orbtrim(fm: ModelFlame):
    """
    Zero out tm_xkick and tm_ykick for elements of type 'orbtrim' in the ModelFlame.

    Parameters:
    - fm (ModelFlame): ModelFlame instance to update.
    """
    iorbtrim = fm.find(type='orbtrim')
    for i in iorbtrim:
        elem = fm.get_element(index=i)[0]['properties']
        if "tm_xkick" in elem:
            fm.reconfigure(i, {'tm_xkick': 0.})
        if "tm_ykick" in elem:
            fm.reconfigure(i, {'tm_ykick': 0.})


def update_zL(fm: ModelFlame):
    """
    Update the z and L attributes for elements.

    Parameters:
    - fm (ModelFlame): ModelFlame instance to update.

    Returns:
    - ModelFlame: Updated ModelFlame instance.
    """
    i = 0
    z = 0
    while True:
        try:
            elem = fm.get_element(index=i)[0]
            elem_prop = elem['properties']
        except IndexError:
            break
        if 'L' in elem_prop:
            L = elem_prop['L']
        else:
            L = 0
        fm.reconfigure(i, {'z': z, 'L': L})
        z += L
        i += 1


def get_df_by_type(type: str, fm: ModelFlame,
                   from_Dnum = None,
                   to_Dnum = None,
                   ) -> pd.DataFrame:
    """
    Get a DataFrame containing properties of elements of the given type from the ModelFlame.

    Parameters:
    - type (str): Element type to retrieve.
    - fm (ModelFlame): ModelFlame instance to retrieve elements from.

    Returns:
    - pd.DataFrame: DataFrame containing properties of elements of the specified type.
    """
    from_Dnum = from_Dnum or 0
    to_Dnum   = to_Dnum   or 99999
    
    
    ind = fm.get_index_by_type(type)[type]
    elems = fm.get_element(index=ind)
    elem_props = []
    for i, elem in enumerate(elems):
        elem_props.append(elem['properties'])
        elem_props[-1]['index'] = ind[i]
        
    df = pd.DataFrame(copy(elem_props)).set_index('index')
    names = sort_by_Dnum([pv for pv in df['name'].values if (from_Dnum <= get_Dnum_from_pv(pv) <= to_Dnum)])
    df = df[df['name'].isin(names)]
    return df



def plot_lattice(fm,ax,
                 starting_offset=0,
#                  start=None,
#                  end=None,
                 xmin=None,
                 xmax=None,
                 ymin=None,
                 ymax=None,
                 legend=True,
                 legend_ncol=2,
                ):
    pl = _plot_lattice(fm)
    xmin_, xmax_ = ax.get_xlim()
    if xmin is None:
        xmin = xmin_
    if xmax is None:
        xmax = xmax_
    
    ymin_, ymax_ = ax.get_ylim()
    if ymin is None:
        ymin = ymin_
    if ymax is None:
        ymax = ymax_
    
    pl(ax=ax,
       starting_offset=starting_offset,
       start=None,
       end=None,
       ymin=ymin,
       ymax=ymax,
       legend=legend,
       legend_ncol=legend_ncol,
      )
    ax.set_xlim(xmin,xmax)

    
    
class _plot_lattice():
    def __init__(self, fm: ModelFlame):
        """
        Initialize the plot_lattice class with a ModelFlame instance.

        Parameters:
        - M: ModelFlame instance.
        """
        self.M = fm.machine
        self.types = {
            'rfcavity': {'flag': True, 'name': 'rfcavity', 'color': 'orange', 'scale': 0.0},
            'solenoid': {'flag': True, 'name': 'solenoid', 'color': 'red', 'scale': 0.0},
            'quadrupole': {'flag': True, 'name': 'quad', 'color': 'purple', 'scale': 0.0},
            'sextupole': {'flag': True, 'name': 'sext', 'color': 'navy', 'scale': 0.0},
            'sbend': {'flag': True, 'name': 'bend', 'color': 'green', 'scale': 0.0},
            'equad': {'flag': True, 'name': 'e-quad', 'color': 'blue', 'scale': 0.0},
            'edipole': {'flag': True, 'name': 'e-dipole', 'color': 'lime', 'scale': 0.0},
            'bpm': {'flag': True, 'name': 'bpm', 'color': 'm', 'scale': 0.0},
            'orbtrim': {'flag': True, 'name': 'corr', 'color': 'black', 'scale': 0.0},
            'stripper': {'flag': True, 'name': 'stripper', 'color': 'y', 'scale': 0.0},
            'marker': {'flag': True, 'name': 'pm', 'color': 'c', 'scale': 0.0}
        }

    def _get_scl(self, elem):
        """
        Get arbitrary strength of the optical element.

        Parameters:
        - elem: Optical element properties.

        Returns:
        - float: Arbitrary strength of the optical element.
        """
        try:
            if elem['type'] == 'rfcavity':
                scl = elem['scl_fac'] * np.cos(2.0 * np.pi * elem['phi'] / 360.0)
            elif elem['type'] == 'solenoid':
                scl = elem['B']
            elif elem['type'] == 'quadrupole':
                scl = elem['B2'] if 'B2' in elem else 1.0
            elif elem['type'] == 'sextupole':
                scl = elem['B3']
            elif elem['type'] == 'sbend':
                scl = elem['phi']
            elif elem['type'] == 'edipole':
                scl = elem['phi']
            elif elem['type'] == 'equad':
                if hasattr(elem, 'V'):
                    scl = elem['V'] / elem['radius'] ** 2.0
                elif hasattr(elem, 'scl_fac0'):
                    scl = elem['scl_fac0'] / elem['radius'] ** 2.0
                else:
                    scl = 1.0
        except Exception:
            scl = 0.0

        return scl

    def __call__(self,
                 starting_offset=0,
                 start=None,
                 end=None,
                 ymin=0.0,
                 ymax=1.0,
                 legend=True,
                 legend_ncol=2,
                 ax=None):
        """
        Plot the lattice diagram.

        Parameters:
        - starting_offset: Starting position offset.
        - start: Starting index of elements.
        - end: Ending index of elements.
        - ymin: Minimum y-value for the plot.
        - ymax: Maximum y-value for the plot.
        - legend: Whether to display the legend
        """
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,3))
        
        ydif = ymax - ymin
        yscl = ydif
        if ydif == 0.0:
            ydif = ymax*0.1 if ymax != 0.0 else 0.1
            yscl = ydif*0.2
        ycen=ymin-0.2*ydif
        yscl=0.1*yscl
        
        pos = starting_offset
        bp = ycen
        indexes = range(len(self.M))[start:end]
        foundelm = []

        for i in indexes:
            elem = self.M.conf(i)
            try:
                dL = elem['L']
            except:
                dL = 0.0

            if elem['type'] in self.types.keys():
                info = self.types[elem['type']]

                if foundelm.count(elem['type']) == 0:
                    foundelm.append(elem['type'])
                    if legend and info['flag']:
                        ax.fill_between([0,0],[0,0],[0,0], color=info['color'], label=info['name'])

                if info['flag']:
                    if dL != 0.0:
                        bpp = bp
                        if info['scale'] != 0.0:
                            ht = yscl*self._get_scl(elem)/info['scale'] + 0.05
                        else:
                            ht = yscl*np.sign(self._get_scl(elem))

                        if elem['type'] == 'rfcavity' or elem['type'] == 'solenoid':
                            bpp = bp-yscl*0.7
                            ht = yscl*2.0*0.7

                        ax.add_patch(ptc.Rectangle((pos, bpp), dL, ht,
                                                           edgecolor='none',facecolor=info['color']))
                    else:
                        ax.add_line(lin.Line2D([pos,pos],[-yscl*0.3+bp, yscl*0.3+bp],color=info['color']))

            pos += dL

        ax.add_line(lin.Line2D([0.0, pos], [bp,bp], color='gray', zorder=-5))

        if legend:
            ax.legend(ncol=legend_ncol, loc='upper left', bbox_to_anchor=(1.01,0.99))
            
            
            
def noise2twiss(
    x: np.ndarray,
    xalpha: float = 0, xbeta: float = 4, xnemit: float = 0.12,
    yalpha: float = 0, ybeta: float = 4, ynemit: float = 0.12,
    cxy: float = 0, cxyp: float = 0, cxpy: float = 0, cxpyp: float = 0)-> OrderedDict:
    """
    Map noise parameters to twiss parameters.

    Parameters:
    - x (np.ndarray): Array of noise parameters.
    - xalpha (float): Initial xalpha value.
    - xbeta (float): Initial xbeta value.
    - xnemit (float): Initial xnemit value.
    - yalpha (float): Initial yalpha value.
    - ybeta (float): Initial ybeta value.
    - ynemit (float): Initial ynemit value.
    - cxy (float): Initial cxy value.
    - cxyp (float): Initial cxyp value.
    - cxpy (float): Initial cxpy value.
    - cxpyp (float): Initial cxpyp value.

    Returns:
    OrderedDict: Twiss parameters.

    This function maps noise parameters to twiss parameters using predefined formulas.
    """
    if len(x)==6:
        tmp = np.zeros(10)
        tmp[:6]=x[:]
        x = tmp
    twiss = {
        'xalpha': xalpha + 0.5*x[0],
        'xbeta' : xbeta * np.exp(x[1]*0.4),
        'xnemit': xnemit* np.exp(x[2]*0.2),
        'yalpha': yalpha + 0.5*x[3],
        'ybeta' : ybeta * np.exp(x[4]*0.4),
        'ynemit': ynemit* np.exp(x[5]*0.2),
        'cxy'   : np.tanh(0.5*x[6]+ np.arctanh(cxy  )),
        'cxyp'  : np.tanh(0.5*x[7]+ np.arctanh(cxyp )),
        'cxpy'  : np.tanh(0.5*x[8]+ np.arctanh(cxpy )),
        'cxpyp' : np.tanh(0.5*x[9]+ np.arctanh(cxpyp)),
    }
    return twiss


def noise2bmstate(bmstate, 
                  x,
                  xalpha: float = 0, xbeta: float = 4, xnemit: float = 0.12,
                  yalpha: float = 0, ybeta: float = 4, ynemit: float = 0.12,
                  cxy: float = 0, cxyp: float = 0, cxpy: float = 0, cxpyp: float = 0):
    twiss = noise2twiss(x, xalpha, xbeta, xnemit, yalpha, ybeta, ynemit, cxy, cxyp, cxpy, cxpyp)
    
#     display(twiss)
    
    bmstate.set_twiss('x',alpha=twiss['xalpha'],beta=twiss['xbeta'],nemittance=twiss['xnemit'])
    bmstate.set_twiss('y',alpha=twiss['yalpha'],beta=twiss['ybeta'],nemittance=twiss['ynemit'])
    bmstate.set_couple('x' ,'y' ,twiss['cxy'])
    bmstate.set_couple('x' ,'yp',twiss['cxyp'])
    bmstate.set_couple('xp','y' ,twiss['cxpy'])
    bmstate.set_couple('xp','yp',twiss['cxpyp'])
    
    
def from_flame_evals_to_machine_evals(flame_evals_df):
    data = {}
    for name,field in flame_evals_df.columns:
        PV = get_PVname_from_FLAMEname(name,fm_field=field,tag='RD')
        data[PV] = [convert.fm2mp(val,name,fm_field=field) for val in flame_evals_df[(name,field)] ]  
    return pd.DataFrame(data)


def from_machine_evals_to_flame_evals(machine_evals_df,fm):
    machine_names = machine_evals_df.columns.tolist()
    flame_elements = get_FMelem_from_PVs(machine_names, fm=fm)
    info = {elem['properties']['name']:{'index':elem['index']} 
            for elem in flame_elements}
    
    fm_fields = []
    mp_fields = []
    for PV in machine_names:
        polarity_, mp_elem, mp_field_, fm_field_ = convert._get_conversion_info_from_PV(PV)
        machine_name, mp_field = split_name_field_from_PV(PV)
        i_ = mp_field.find('_')
        if i_ != -1:
            mp_field = mp_field[:i_]
        if not mp_field_ == mp_field:
            raise ValueError(f'field from PV {PV} is {mp_field} but field from conversion pre-set {mp_field_}. FLAME field {fm_field_} may not right')
        mp_fields.append(mp_field )
        fm_fields.append(fm_field_)
    
    df = {}
    for fm_name, PV, fm_field, mp_field in zip(info.keys(), machine_names, fm_fields, mp_fields):
        df[(fm_name,fm_field)] = []
        for ind,rowval in machine_evals_df.iterrows():
#             print("fm_name, PV, fm_field, mp_field",fm_name, PV, fm_field, mp_field)
            df[(fm_name,fm_field)].append(
                convert.mp2fm(rowval[PV],PV,fm_field=fm_field,mp_field=mp_field) 
            )
    df = pd.DataFrame(df)
    return {'info':info,
            'df'  :df}


def evaluate_flame_evals(
    flame_evals, 
    fm, 
    from_bmstate = None,
    from_element = None,
    to_element = None,
    monitor_indices = None,
    monitor_names = None,
    collect_data_args = None,
    restore_configure = False,
    ) -> float:
    """
    Evaluate FLAME evaluation results 
    """   
    if monitor_indices is None and monitor_names is None:
        raise ValueError(f"monitor_indices or monitor_names must be provided")
    elif monitor_indices is None:
        monitor_dict = fm.get_index_by_name(monitor_names)
        monitor_indices = []
        for name in monitor_names:
            monitor_indices.append(monitor_dict[name][0])
        argsorted = np.argsort(monitor_indices)
        monitor_indices = [monitor_indices[i] for i in argsorted]
        monitor_names   = [monitor_names[i] for i in argsorted]
    elif monitor_names is None:
        monitor_names = [elem['properties']['name'] for elem in fm.get_element(index=monitor_indices)]
        monitor_indices = np.sort(monitor_indices).tolist()
    else:
        argsorted = np.argsort(monitor_indices)
        monitor_indices = [monitor_indices[i] for i in argsorted]
        monitor_names   = [monitor_names[i] for i in argsorted]

  
    if restore_configure:
        _restore_configs = [(flame_evals['info'][name]['index'], 
                            {field: fm.get_element(index=flame_evals['info'][name]['index'])[0]['properties'][field]})
                           for name,field in flame_evals['df'].columns]
    calQ = False   
    if collect_data_args is None:
        collect_data_args = ["xrms", "yrms", "cxy"]
    if "Q" in collect_data_args:
        calQ = True
        collect_data_args_tmp = list(set(list(collect_data_args) + ["xrms","yrms"]))
        collect_data_args_tmp.remove("Q")
    else:
        collect_data_args_tmp = copy(collect_data_args)
    fm_results = {(m,arg):[] for m in monitor_names for arg in collect_data_args}
      
    for loc,flame_eval in flame_evals['df'].iterrows():
        #print("loc,flame_eval",loc,flame_eval)
        for name_field, value in flame_eval.items():
            elem_name  = name_field[0]
            elem_field = name_field[1]
            elem_index = flame_evals['info'][elem_name]['index']
            #print("elem_index, {elem_field: value}",elem_index, {elem_field: value})
            fm.reconfigure(elem_index, {elem_field: value})
        
        r, s = fm.run(bmstate = from_bmstate,
                      monitor = monitor_indices, 
                      from_element = from_element,
                      to_element   = to_element)
        r = fm.collect_data(r,*collect_data_args_tmp)
      
        if calQ:
            r["Q"] = r['xrms']**2 -  r['yrms']**2

        for i,monitor in enumerate(monitor_names):
            for arg in collect_data_args: 
                fm_results[(monitor,arg)].append(r[arg][i])
           
    if restore_configure:
        for conf in _restore_configs:
            fm.reconfigure(conf[0], conf[1])
        
    return pd.DataFrame(fm_results)



def calculate_loss_from_flame_evals_goals(
    flame_evals, 
    flame_goals,
    fm, #flame_utils.ModelFlame, 
    from_bmstate = None,
    from_element = None,
    to_element = None,
    return_flame_sim_result = False,
    restore_configure = False,
    normalization_factor = None
    ) -> float:
    """
    Evaluate FLAME evaluation results and goals.
    This function evaluates FLAME evaluation results against specified goals and calculates a loss value.
    The loss is computed based on the differences between the achieved and target values for specified FLAME elements.
    """  
    collect_data_args = flame_goals['df'].columns.get_level_values(1).unique().to_list()
    normalization_factor = normalization_factor or {k:1 for k in collect_data_args}
    assert set(collect_data_args) >= set(normalization_factor.keys())

    monitor_indices = []
    monitor_names   = []
    for name, info in flame_goals['info'].items():
        monitor_indices.append(info['index'])
        monitor_names.append(name)

    fm_results = evaluate_flame_evals(
        flame_evals,fm,
        from_bmstate = from_bmstate,
        from_element = from_element,
        to_element = to_element,
        monitor_indices = monitor_indices,
        monitor_names = monitor_names,
        collect_data_args = collect_data_args,
        restore_configure = restore_configure)
                                      
    loss = ((fm_results - flame_goals['df'])**2).mean()    
    # loss = ((fm_results-flame_goals['df'])**2/(flame_goals['df'].abs()+4)**2).mean()
    for monitor,bmstat in flame_goals['df'].columns:
        loss[(monitor,bmstat)] /= (normalization_factor[bmstat]**2)
    
    if return_flame_sim_result:
        return fm_results
    else:
        return loss.mean()


def fit_moment1(fm, 
                flame_evals,
                flame_goals,
                from_bmstate = None,
                from_element = None,
                to_element = None,
                n_try = 5, 
                stop_criteria = 0.5,
                start_fit_from_current_bmstate = True,
                plot_fitting_quality = False,
               ) -> float:
    """
    Fit initial rms beam moments

    Parameters:
    - fm (Any): FLAME machine object.
    
    Returns:
    float: Loss value.
    """
    from_bmstate = from_bmstate or fm.bmstate
    if start_fit_from_current_bmstate:
        xalpha, xbeta, _ = copy(from_bmstate.get_twiss('x'))
        yalpha, ybeta, _ = copy(from_bmstate.get_twiss('y'))
        xnemit = copy(from_bmstate.xnemittance)
        ynemit = copy(from_bmstate.ynemittance)
        cxy, cxyp, cxpy, cxpyp = copy([from_bmstate.couple_xy, from_bmstate.couple_xyp, from_bmstate.couple_xpy, from_bmstate.couple_xpyp])
    else:
        xalpha, xbeta, xnemit = 0.0, 4.0, 0.12
        yalpha, ybeta, ynemit = 0.0, 4.0, 0.12
        cxy, cxyp, cxpy, cxpyp= 0, 0, 0, 0
    
    _restore_configs = [(flame_evals['info'][name]['index'], 
                        {field: fm.get_element(index=flame_evals['info'][name]['index'])[0]['properties'][field]})
                       for name,field in flame_evals['df'].columns]

    def loss_ftn(x: np.ndarray) -> float:
        noise2bmstate(from_bmstate, x,
                      xalpha, xbeta, xnemit, yalpha, ybeta, ynemit, cxy, cxyp, cxpy, cxpyp)

        loss = calculate_loss_from_flame_evals_goals(flame_evals, flame_goals, fm, 
                                          from_bmstate = from_bmstate, 
                                          from_element = from_element, 
                                          to_element = to_element, 
                                          )
        # reg_loss = max(from_bmstate.xnemittance/from_bmstate.ynemittance,from_bmstate.ynemittance/from_bmstate.xnemittance)
        # reg_loss = (max(reg_loss,1.5) - 1.5)**2
      
        return loss# + 0.01*reg_loss

    result = NelderMead(loss_ftn, np.zeros(10),
                        simplex_size = 0.1, 
                        bounds = None, 
                        tol = min(1e-4, stop_criteria*0.1) )
    best_loss = result.fun
    best_result = result
    x0 = []
    i=0
    for i in range(n_try-1):
        if best_loss < stop_criteria:
                break
        if len(x0) == 0:
            x0.append(result.x)
#         elif np.all(np.mean((np.array(x0) - result.x.reshape(1,-1))**2,axis=1) > 0.01):
#             x0.append(result.x.copy())
        else:
            x0.append(np.random.randn(10))
        print(f"{i}-th trial, current_loss: {result.fun:.3g}, best_loss: {best_loss:.3g}")#, x0: {np.array2string(x0[-1], precision=3, separator=',')}")
        result = NelderMead(loss_ftn, x0[-1].copy(),
                            simplex_size = 0.1, 
                            bounds = None, 
                            tol = min(1e-4, stop_criteria*0.1))
        if result.fun < best_loss:
            best_result = result
            best_loss = result.fun
            
    print(f"{i+1}-th trial, current_loss: {result.fun:.3g}, best_loss: {best_loss:.3g}")
    
    for conf in _restore_configs:
        fm.reconfigure(conf[0], conf[1])
    
    if best_loss > 4:
        warn(f'fit was not successful. loss: {best_loss}. ')
        
        
    noise2bmstate(from_bmstate, best_result.x,
                  xalpha, xbeta, xnemit, yalpha, ybeta, ynemit, cxy, cxyp, cxpy, cxpyp)
                  
    fit_loss = calculate_loss_from_flame_evals_goals(  
                                        flame_evals,
                                        flame_goals,
                                        fm, from_bmstate,
                                        from_element = from_element,
                                        to_element = to_element,
                                        return_flame_sim_result=False,
                                        restore_configure=True)
    
    if plot_fitting_quality:
        flame_results = calculate_loss_from_flame_evals_goals(  
                                                flame_evals,
                                                flame_goals,
                                                fm, from_bmstate,
                                                from_element = from_element,
                                                to_element = to_element,
                                                return_flame_sim_result=True,
                                                restore_configure=True)
        monitors = list(flame_goals['info'].keys())
        for i,m in enumerate(monitors):
            fig,ax = plt.subplots(figsize=(5,2), dpi=96)
            for k in flame_results[m].columns:
                col = (m,k)
                if not np.all(np.isnan(flame_goals['df'][col])):
                    x = np.arange(len(flame_results))
                    ax.scatter(x,flame_goals['df'][col],s=4)
                    ax.plot   (x,flame_results[col],label=k)
            icolon = m.find(":")
            if icolon > 0:
                m_short = m[icolon+1:]
            ax.set_title('meas VS fit, '+m_short)
            ax.legend()
        ax.set_xlabel('data no.')
        
    return fit_loss

        
        
def make_FALEM_evals_or_goals(fm,tuples_of_fm_elem_names_n_fields=None,values=None,df=None,
                              ):
    if df is None:
        assert tuples_of_fm_elem_names_n_fields is not None
        assert values is not None
        fm_elem_names = list(set([name for name,field in tuples_of_fm_elem_names_n_fields]))
        values = np.atleast_2d(values)
    else:
        fm_elem_names = sort_by_Dnum(df.columns.get_level_values(0).unique().tolist())
        df = df[fm_elem_names]
        tuples_of_fm_elem_names_n_fields = df.columns.tolist()
        values = df.values
    assert len(tuples_of_fm_elem_names_n_fields) == values.shape[1]
    
    info = {elem['properties']['name']:{'index':elem['index']} for elem in fm.get_element(name=fm_elem_names)}
    
    if df is None:
        df = {}
        icol = 0
        for name,field in tuples_of_fm_elem_names_n_fields:
            df[(name,field)] = values[:,icol]
            icol +=1
        df = pd.DataFrame(df)

    return {
        'info': info,
        'df':df
    }