import re
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import optimize

import io
import contextlib
import warnings

def _warn(message, *args, **kwargs):
    return 'warning: ' +str(message) + '\n'
#     return _warn(x,stacklevel=2)  

warnings.formatwarning = _warn
def warn(x):
    return warnings.warn(x)


@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

_name_conversions =(
    (':PSQ_' , ':Q_'  ),
    (':PSQ_' , ':QV_' ),
    (':PSQ_' , ':QH_' ),
    (':PSC2_', ':DCH_'),
    (':PSC1_', ':DCV_'),
)

            
def NelderMead(loss_ftn, x0, simplex_size: float = 0.05, 
                             bounds: Optional[List[Tuple[float, float]]] = None, 
                             tol: float = 1e-4):
    """
    Perform optimization using the Nelder-Mead method.

    Parameters:
    - loss_ftn (callable): The objective function to be minimized.
    - x0 (array-like): Initial guess for the solution.
    - simplex_size (float): Initial step size for each dimension. Default is 0.15.
    - bounds (list or None, optional): Bounds for variables. If None, no bounds are applied.
        Otherwise, it should be a list of tuples, each containing the lower and upper bounds for each dimension.
        Default is None.

    Returns:
    - result (scipy.optimize.OptimizeResult): The optimization result.

    Notes:
    - If bounds are provided, the initial simplex is adjusted to ensure it lies within the specified bounds.
    - The optimization result is returned as an instance of `scipy.optimize.OptimizeResult`.
    """
    n = len(x0)
    initial_simplex = np.vstack([x0] * (n + 1))

    if bounds is None:
        for i in range(n):
            initial_simplex[i + 1, i] += simplex_size
    else:
        bounds = np.array(bounds)
        assert np.all(x0 <= bounds[:, 1]) and np.all(bounds[:, 0] <= x0)
        for i in range(n):
            dx = simplex_size * (bounds[i, 1] - bounds[i, 0])
            if x0[i] + dx > bounds[i, 1]:
                initial_simplex[i + 1, i] -= dx
            else:
                initial_simplex[i + 1, i] += dx

            initial_simplex[i + 1, :] = np.clip(initial_simplex[i + 1, :], a_min=bounds[:, 0], a_max=bounds[:, 1])

    result = optimize.minimize(loss_ftn, x0, method='Nelder-Mead', bounds=bounds, tol=tol,
                               options={'initial_simplex': initial_simplex})

    return result


def is_list_of_lists(input_list):
    if not isinstance(input_list, list):
        return False
    
    for item in input_list:
        if not isinstance(item, list):
            return False
    
    return True


def from_listdict_to_pd(data):
    # Extract all keys from the list of dicts
    all_keys = set().union(*data)

    # Create a dictionary of lists
    dict_of_lists = {}

    # Populate the dictionary of lists
    for key in all_keys:
        dict_of_lists[key] = []
        for d in data:
            dict_of_lists[key].append(d.get(key, np.nan))

    # Ensure all lists have the same length
    max_length = max(len(lst) for lst in dict_of_lists.values())
    for key in dict_of_lists:
        dict_of_lists[key] += [np.nan] * (max_length - len(dict_of_lists[key]))

    return pd.DataFrame(dict_of_lists)


def get_Dnum_from_pv(pv: str) -> int or None:
    """
    Extracts the D number from a PV string.
    Args:
        pv (str): The PV string.
    Returns:
        int or None: The extracted D number or None if not found.
    """
    try:
        match = re.search(r"_D(\d{4})", pv)
        if match:
            return int(match.group(1))
        else:
            return None
    except AttributeError:
        return None
    

def split_name_field_from_PV(PV: str, 
                           return_device_name: bool =True) -> tuple:
    """
    Splits the PV into name and key components.

    Args:
        PV (str): The PV string.

    Returns:
        tuple: A tuple containing the name and key components.
    """
    # Find the index of the first colon
    first_colon_index = PV.find(':')

    if first_colon_index == -1:
        print(f"Name of PV: {PV} is not found")
        return None, None
    
    if return_device_name:
        for dev_name, phys_name in _name_conversions:
            PV = PV.replace(phys_name,dev_name)

    second_colon_index = PV.find(':', first_colon_index + 1)
    if second_colon_index != -1:
        return PV[:second_colon_index], PV[second_colon_index + 1:]
    else:
        return PV, None

    
def sort_by_Dnum(strings):
    """
    Sort a list of PVs by dnum.
    """
    # Define a regular expression pattern to extract the 4-digit number at the end of each string
    pattern = re.compile(r'\D(\d{4})$')

    # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
    def sorting_key(s):
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        return 0  # Default value if no match is found

    # Sort the strings based on the custom sorting key
    sorted_strings = sorted(strings, key=sorting_key)
    return sorted_strings


def post_process_BPMdf(df,from_Dnum,to_Dnum,fill_NaN_for_large_MAG_err=True,index=0,remove_TISRAW=False,remove_CURRENT=False):
    """
    This function performs filtering, sorting, and additional calculations on the input DataFrame.
    It filters data based on Dnum, sorts it, calculates mean, error, and other derived quantities,
    Finally, it calculates BPM-Q, BPM-Qlogratio, normalized 4 pickups and standard mean error of them.

    Parameters:
        df (DataFrame): DataFrame containing BPM data.
        from_Dnum (int): Starting Dnum value for filtering.
        to_Dnum (int): Ending Dnum value for filtering.
        index (int, optional): Index for the resulting DataFrame. Defaults to 0.

    Returns:
        DataFrame: Post-processed DataFrame containing BPM data.
    """
    # filter out by Dnum
    Dnum_filter = sort_by_Dnum([pv for pv in df.index if from_Dnum <= get_Dnum_from_pv(pv) <= to_Dnum])
    df = df.loc[Dnum_filter]
    if remove_TISRAW:
        locs = []
        for pv in df.index:
            if ':TISRAW' in pv:
                continue
            locs.append(pv)
        df = df.loc[locs]
    if remove_CURRENT:
        locs = []
        for pv in df.index:
            if ':CURRENT' in pv:
                continue
            locs.append(pv)
        df = df.loc[locs]
    
    # sort
    BPMnames = []
    Keys = []
    for pv in df.index:
        bpm_name, key = split_name_field_from_PV(pv)
        BPMnames.append(bpm_name)
        Keys.append(key)
    BPMnames = sort_by_Dnum(list(np.unique(BPMnames)))
    Keys     = list(np.unique(Keys    ))
    Keys.sort()
    sorted_index = []
    for bpm in BPMnames:
        for key in Keys:
            sorted_index.append(bpm+':'+key)
    df = df.loc[sorted_index]
    
    assert np.all(df['mean'].to_frame().T.columns == df.index)
            
    multi_index = []
    err_multi_index = []
    for pv in df.index:
        bpm_name, key = split_name_field_from_PV(pv)
        multi_index.append((bpm_name, key.replace('TISMAG161_', 'U').replace('_RD', '')))
        err_multi_index.append((bpm_name, key.replace('TISMAG161_', 'U').replace('_RD', '')+'_err'))
    multi_index = pd.MultiIndex.from_tuples(multi_index)
    err_multi_index = pd.MultiIndex.from_tuples(err_multi_index)
    BPMnames = np.unique([bpm_name for bpm_name, _ in multi_index])
    
    mean = df['mean'].to_frame().T
    mean.columns = multi_index
    mean.index = [index]
    
    mean_err =  (df['std']/df['#']**0.5).to_frame().T
    mean_err.columns = err_multi_index
    mean_err.index = [index]

    Qterms_val = []
    Qterms_col = []
    for bpm_name in BPMnames:
        U = [float((mean[bpm_name]['U' + str(iv)]).iloc[0]) for iv in range(1, 5)]
        # assert np.all(np.array(U) > 0)
        if np.any(np.array(U) < 0):
            print(f"{bpm_name} U is negative. U: {U}")
            print(f"{bpm_name} U is negative. U_err: {U_err}")
        U_err = [float((mean_err[bpm_name]['U' + str(iv) +'_err']).iloc[0]) for iv in range(1, 5)]
        # assert np.all(np.array(U_err) > 0)
        if np.any(np.array(U_err) < 0):
            print(f"{bpm_name} U_err is negative. U: {U}")
            print(f"{bpm_name} U_err is negative. U_err: {U_err}")
            
        
        if np.any(np.array(U_err)/np.array(U)) > 1:
            print(f"{bpm_name} signal noise large than signal strength. U_err[i]/U[i] are: {np.array(U_err)/np.array(U)}")
            
        Qterms_col.append((bpm_name, 'Q'))
        Qterms_col.append((bpm_name, 'Q_err'))
        Qterms_col.append((bpm_name, 'Q_logratio'))
        Qterms_val.append((U[1] + U[2] - (U[0] + U[3])) / np.sum(U))
        Qterms_val.append( 2*(  (U_err[0]*(U[1]+U[2]))**2  
                               +(U_err[1]*(U[0]+U[3]))**2
                               +(U_err[2]*(U[0]+U[3]))**2
                               +(U_err[3]*(U[1]+U[2]))**2 )**0.5
                            / np.sum(U)**2)
        Qterms_val.append(np.log(U[1]*U[2] / (U[0]*U[3]) ))
#         for iv in range(1, 5):
#             mean.loc[:, (bpm_name, 'U' + str(iv))] /= np.sum(U)
#             mean_err.loc[:, (bpm_name, 'U' + str(iv) + '_err')] /= np.sum(U)
    Qterms=pd.DataFrame(np.array(Qterms_val).reshape(1,-1), columns=pd.MultiIndex.from_tuples(Qterms_col), index=[index])
    
    
    df = pd.concat((mean, mean_err, Qterms), axis=1)[sort_by_Dnum(BPMnames)]
    if fill_NaN_for_large_MAG_err:
#         pd.set_option('future.no_silent_downcasting', True) # suppress warning regarding ffill()
        for name in BPMnames:
            # Standard mean error of BPM_MAG should be less than 5%
            mask = df[name]['MAG_err'] > 0.05 * df[name]['MAG']
            df.loc[mask, name] = df[name][mask].ffill()
#     pd.set_option('future.no_silent_downcasting', False)
    return df


def fill_NaN_for_suspicious_BPMdata_based_on_MAG(df):
    BPMnames = sort_by_Dnum(df.columns.get_level_values(0).unique().tolist())
#     pd.set_option('future.no_silent_downcasting', True) # suppress warning regarding ffill()
    df = df.copy()
    
    for name in BPMnames:
        # Standard mean error of BPM_MAG should be less than 5%
        mask = df[name]['MAG_err'] > 0.05 * df[name]['MAG']
        df.loc[mask, name] = df[name][mask].ffill()
        if np.any(mask):
            print(name, 'removed data fraction due to large MAG noise:', 1 -mask.sum() / len(mask))

    mag0 = df[(BPMnames[0],'MAG')]
    for name in BPMnames[1:]:
        # Beam loss: BPM_MAG (+mean_err) should not be less than 90% of 0.9 quantile over samples
        mask = (df[(name,'MAG')] +df[(name,'MAG_err')])/mag0 < 0.95 * (df[(name,'MAG')]/mag0).quantile(0.95)
        df.loc[mask, name] = df[name][mask].ffill()
        if np.any(mask):
            print(name, 'removed data fraction due to suspected bema loss:', 1 -mask.sum() / len(mask))
        
    return df


def is_BPMdf_consistent(df, should_consistent_upto_Dnum, 
                        mean_ref=None, 
                        tolerance=None, 
                        verbose=False,
                        fill_NaN_inconsistent_row = False,
                       ):
    
    BPMnames = sort_by_Dnum(list(set(df.columns.get_level_values(0))))
    BPMnames = [name for name in BPMnames if get_Dnum_from_pv(name) <= should_consistent_upto_Dnum ]

    if mean_ref is None:
        mean_ref = df.mean()
    else:
        mean_ref = mean_ref[[(name, k) for name in BPMnames for k in ['XPOS', 'YPOS', 'MAG', 'Q']]]
    BPMnames = [name for name in BPMnames if not np.any([np.isnan(mean_ref[(name,k)]) for k in ['XPOS', 'YPOS', 'MAG', 'Q']])]
    mean_ref = mean_ref[[(name, k) for name in BPMnames for k in ['XPOS', 'YPOS', 'MAG', 'Q']]]

    df_err = df[[(name, k) for name in BPMnames for k in ['XPOS_err', 'YPOS_err', 'MAG_err', 'Q_err']]]
    df_ = df[[(name, k) for name in BPMnames for k in ['XPOS', 'YPOS', 'MAG', 'Q']]]

    if tolerance is None:
        tolerance = {'XPOS':0.2,
                     'YPOS':0.2,
                     'Q'   :0.1,
                     }
    else:
        tolerance = {k:tolerance[k] for k in ['XPOS', 'YPOS', 'Q']}
    tolerance = {(name, k):[tolerance[k]] for name in BPMnames for k in ['XPOS', 'YPOS', 'Q']}
    for name in BPMnames:
        tolerance[(name,'MAG')]  = [mean_ref[(name,'MAG')]+1e-10]

    tolerance = pd.DataFrame(tolerance)
    tolerance_broadcasted = pd.concat([tolerance] * len(df_), ignore_index=True)

    for name in BPMnames:
        for k in ['XPOS', 'YPOS', 'Q','MAG']:
            tolerance_broadcasted[(name,k)] = (tolerance_broadcasted[(name,k)]**2 
                                              +4**2*df_err[(name,k+'_err')]**2)**0.5


    consistency_df = (df_ - mean_ref).abs()/tolerance_broadcasted < 1

    consistent = True
    for loc,val in consistency_df.iterrows():
        if val.sum()/len(val) < 0.95:
            consistent = False
            if fill_NaN_inconsistent_row:
                df.loc[loc] = np.nan
                display(df.loc[[loc-1,loc]])
   
    return consistent


def get_consistent_row_from_two_BPMdf(df_test, df_ref, should_consistent_upto_Dnum, 
                                      fill_NaN_inconsistent_row_of_df_test=False):
    if not is_BPMdf_consistent(df = df_ref, 
                               should_consistent_upto_Dnum = should_consistent_upto_Dnum,
                               verbose = True):
        raise ValueError('BPMdf_ref is not consistent')
    BPMnames = set(df_ref.columns.get_level_values(0))
    assert BPMnames == set(df_test.columns.get_level_values(0))
    BPMnames = sort_by_Dnum(list(BPMnames))
    consistent_irow = np.array([True] * len(df_test))
    for name in BPMnames:
        if get_Dnum_from_pv(name) > should_consistent_upto_Dnum:
            break
        mean = df_ref[name].loc[:, ['XPOS', 'YPOS', 'MAG', 'Q']].mean()
        for k in ['XPOS', 'YPOS']:
            if np.isnan(mean[k]):
                continue                
            consistent_irow = np.logical_and(consistent_irow, (mean[k] - df_test[name][k]).abs() < 0.2)
        consistent_irow = np.logical_and(consistent_irow, (1 - df_test[name]['MAG'] / mean['MAG']).abs() < 0.1)
        consistent_irow = np.logical_and(consistent_irow, (mean['Q'] - df_test[name]['Q']).abs() < 0.1)
    if fill_NaN_inconsistent_row_of_df_test:
        df_test.loc[np.logical_not(consistent_irow)] = np.nan
    return consistent_irow


def match_rows_of_two_df(df_a,df_b,cols,list_of_df_a_like=None,list_of_df_b_like=None,tolerances=None,verbose=False):
    if tolerances is None:
        tolerances = pd.Series({col:0.1 for col in cols}) # assuming col is I_RD of FRIB Quads
    elif isinstance(tolerances,dict):
        tolerances = pd.Series(tolerances)
    else:
        assert isinstance(tolerances,pd.Series)
        
    i2j = {}
    for irow,value in df_a[cols].iterrows():
        distance = ((value - df_b[cols])/tolerances).abs().max(axis=1)
        i = np.argmin(distance)
        if distance[i] < 1:
            jrow = df_b.index[i]
            i2j[irow] = jrow
    if verbose:
        print(i2j)

    df_a_new = pd.DataFrame({col:
                             df_a[col].loc[[loc for loc in i2j.keys()]].tolist() +
                             df_a[col].loc[[loc for loc in df_a.index if loc not in i2j.keys()]].tolist() +
                             [np.nan]*(len(df_b)-len(i2j)) 
                             for col in df_a.columns})
    df_b_new = pd.DataFrame({col:df_b[col].loc[[loc for loc in i2j.values()]].tolist() + 
                             [np.nan]*(len(df_a)-len(i2j)) + 
                             df_b[col].loc[[loc for loc in df_b.index if loc not in i2j.values()]].tolist()  
                             for col in df_b.columns})    
    
    list_of_df_a_like_new = None
    if list_of_df_a_like is not None:
        assert isinstance(list_of_df_a_like,list)
        list_of_df_a_like_new = []
        for df in list_of_df_a_like:
            assert isinstance(df,pd.DataFrame) and len(df) == len(df_a)
            df_new = pd.DataFrame({col:
                df[col].loc[[loc for loc in i2j.keys()]].tolist() +
                df[col].loc[[loc for loc in df.index if loc not in i2j.keys()]].tolist() +
                [np.nan]*(len(df_b)-len(i2j)) 
                for col in df.columns})
            list_of_df_a_like_new.append(df_new)
            
    list_of_df_b_like_new = None
    if list_of_df_b_like is not None:
        assert isinstance(list_of_df_b_like,list)
        list_of_df_b_like_new = []
        for df in list_of_df_b_like:
            assert isinstance(df,pd.DataFrame) and len(df) == len(df_b)
            df_new = pd.DataFrame({col:df[col].loc[[loc for loc in i2j.values()]].tolist() + 
                                   [np.nan]*(len(df_a)-len(i2j)) + 
                                   df[col].loc[[loc for loc in df.index if loc not in i2j.values()]].tolist()  
                                   for col in df.columns}) 
            list_of_df_b_like_new.append(df_new)
    
    return df_a_new, df_b_new, list_of_df_a_like_new, list_of_df_b_like_new
    


def datetime_from_Ymd_HMS(xstr):
    datetime_match = re.search(r"\d{8}_\d{6}", xstr)
    if datetime_match:
        return datetime.strptime(datetime_match.group(0), "%Y%m%d_%H%M%S")
    else:
        raise ValueError(f'input straing {xstr} does not constain Ymd_HMS')