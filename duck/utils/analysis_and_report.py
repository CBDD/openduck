#!/usr/bin/env python
"""Analysis of dynamic undocking (DUck) steered MD outputs.

DUck pulls the key hydrogen bond of a protein-ligand complex from its bound
distance (2.5 A by default) to 5.0 A with a harmonic spring that moves at
constant speed, recording the accumulated work along the way. Two descriptors
are derived from those work profiles:

* **WQB** (work to reach the quasi-bound state). For each SMD replica the work
  profile is searched for its first local minimum (the relaxed bound state);
  WQB is the maximum work reached after that minimum minus the minimum
  itself. The WQB of a system is the lowest WQB over all its replicas, i.e. the
  most pessimistic estimate of how robust the hydrogen bond is.
* **dG_QB** (free energy to reach the quasi-bound state), estimated with the
  Jarzynski equality, ``dG = -kT ln <exp(-W/kT)>``, over all replicas of a given
  temperature. Each work profile is first shifted so that its minimum within
  the first half of the profile is zero. Confidence is assessed by
  bootstrapping: replicas are resampled with replacement, the Jarzynski
  profile is recomputed and dG_QB is taken as the maximum after the minimum of
  that profile. The reported value is the mean, standard deviation and
  standard error over the bootstrap samples (all temperatures pooled).

SMD outputs come in two formats, detected automatically by :func:`read_duck_dat`:

* AMBER ``jar=1`` DUMPAVE files (``DUCK_*/duck.dat``) and the current OpenDUck
  OpenMM engine (``duck_runs/smd_<cycle>_<T>.dat``): 4 columns, target
  distance r0 (A), actual distance r (A), force (kcal/mol/A) and accumulated
  work (kcal/mol); 5000 rows per replica.
* Legacy OpenDUck OpenMM files: 10 tokens per row including units, e.g.
  ``0 2.502 A 0.3108 nm -30.3 kcal/(A mol) -0.06 kcal/mol``; 1250 rows.

Runs with independent equilibrations (``--equilibrations``) have one ``EQ_<k>/`` folder per equilibration with
the layouts above inside. Their replicas are pooled, and the bootstrap resamples the equilibrations before the
replicas (see :mod:`duck.utils.equilibrations`), so the errors include the variation between equilibrations.
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import sem, shapiro

from duck.utils.equilibrations import PREFIX as EQUILIBRATION_PREFIX
from duck.utils.equilibrations import cluster_bootstrap_indices, equilibration_of, find_equilibrations

#: Boltzmann constant in kcal/(mol K), as used in the original DUck scripts.
KB_KCAL = 0.0019872

#: Number of points per segment when searching the first local work minimum.
WQB_SEGMENT = 200

_AMBER_TOKENS = 4
_LEGACY_OPENMM_TOKENS = 10


# base functions
def save_pickle(file, object):
    with open(file, 'wb') as fh:
        pickle.dump(object, fh)


def read_pickle(file):
    with open(file, 'rb') as fh:
        obj = pickle.load(fh)
    return obj


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _natural_key(text):
    '''Sort key that orders embedded integers numerically (DUCK_2 < DUCK_10).'''
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', str(text))]


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


# SMD output parsing
def read_duck_dat(file_duck_dat):
    """Read an SMD work report, auto-detecting its format.

    Args:
        file_duck_dat (str): Path to an AMBER-style (4 columns) or legacy
            OpenDUck OpenMM (10 tokens per row) SMD report.

    Returns:
        numpy.ndarray: Array of shape (n_rows, 4) with columns target distance
        r0 (A), actual distance r (A), force (kcal/mol/A) and accumulated
        work (kcal/mol).

    Raises:
        ValueError: If the file is empty or its rows do not match a known format.
    """
    with open(file_duck_dat) as fh:
        rows = [line.split() for line in fh if line.strip()]
    if not rows:
        raise ValueError(f'{file_duck_dat} is empty.')
    n_tokens = len(rows[0])
    if n_tokens == _AMBER_TOKENS:
        columns, distance_scale = (0, 1, 2, 3), 1.0
    elif n_tokens == _LEGACY_OPENMM_TOKENS:
        # index, r0 'A', r 'nm', force 'kcal/(A mol)', work 'kcal/mol'
        columns, distance_scale = (1, 3, 5, 8), 10.0
    else:
        raise ValueError(f'{file_duck_dat} has an unknown SMD report format: {n_tokens} columns per row '
                         f'(expected {_AMBER_TOKENS} for AMBER/OpenDUck or {_LEGACY_OPENMM_TOKENS} for legacy OpenMM).')
    try:
        data = np.array([[float(row[c]) for c in columns] for row in rows if len(row) == n_tokens])
    except ValueError as error:
        raise ValueError(f'{file_duck_dat} contains non-numeric values: {error}') from None
    if len(data) != len(rows):
        raise ValueError(f'{file_duck_dat} has rows with inconsistent number of columns.')
    data[:, 1] *= distance_scale
    return data


# wqb report functions
def get_mols_and_format(data_df, mode='min'):
    '''Writeout an rdkit mol object with the specified WQB or Jarzynski information depending on the mode.
    It uses the ligand.pdb file from each DUck directory.'''
    from rdkit import Chem
    mols = []
    for i, row in data_df.iterrows():
        ligand_file = os.path.join(row['System'], 'ligand.pdb')
        if not os.path.isfile(ligand_file):
            raise OSError(f'Cannot open {ligand_file}; make sure you have the ligands in the duck folders to obtain the report in SDF format.')
        mol = Chem.MolFromPDBFile(ligand_file, proximityBonding=False)
        if not mol:
            raise ValueError(f'Error: Could not load {ligand_file}.')
        mol.SetProp('_Name', row['System'])
        if mode == 'min':
            mol.SetProp('WQB', str(row['WQB']))
        elif mode == 'single':
            data = list(row.values)
            data.pop(0)  # pop system
            mol.SetProp('All WQB', ','.join([str(x) for x in data]))
        elif mode == 'avg':
            mol.SetProp('WQB', str(row['WQB']))
            mol.SetProp('WQB AVG', str(row['Average']))
            mol.SetProp('WQB SD', str(row['SD']))
        elif mode == 'jarzynski':
            mol.SetProp('Jarzynski', str(row['Jarzynski']))
            mol.SetProp('Jarzynski_SD', str(row['Jarzynski_SD']))
            mol.SetProp('Jarzynski_SEM', str(row['Jarzynski_SEM']))
        elif mode == 'all':
            mol.SetProp('WQB', str(row['WQB']))
            mol.SetProp('WQB AVG', str(row['Average']))
            mol.SetProp('WQB SD', str(row['SD']))
            mol.SetProp('Jarzynski', str(row['Jarzynski']))
            mol.SetProp('Jarzynski_SD', str(row['Jarzynski_SD']))
            mol.SetProp('Jarzynski_SEM', str(row['Jarzynski_SEM']))
            data = list(row.values)
            mol.SetProp('All WQB', ','.join([str(x) for x in data[7:]]))
        mols.append(mol)
    return mols


def flatten_wqb_dict(info_dict):
    """
    Flatten a dictionary of Wqb values into a more easily digestible format.

    Parameters:
    info_dict (dict): A dictionary with keys as system names and values as tuples containing
    the minimum Wqb and a list of tuples, where each tuple contains a replica name and its Wqb value.

    Returns:
    dict: A dictionary where the keys are 'System' and the replica names (union over all systems, in natural
    order), and the values are lists of the corresponding Wqb values for each system ('Nan' if a system does not
    have that replica).
    """
    per_system = [dict(values[1]) if len(values) > 1 else {} for values in info_dict.values()]
    replicas = sorted({name for replicas in per_system for name in replicas}, key=_natural_key)
    flat = {'System': list(info_dict.keys())}
    for name in replicas:
        flat[name] = [replicas_wqb.get(name, 'Nan') for replicas_wqb in per_system]
    return flat


def _mean_sd(values):
    """Mean and standard deviation of the replica Wqb values ('Nan' if there are none)."""
    wqbs = [x[1] for x in values] if values else []
    if not wqbs:
        return 'Nan', 'Nan'
    return np.mean(wqbs), np.std(wqbs)


def build_report_df(info_dict, mode='min', sorting_column='System'):
    '''Generate a report DataFrame from the dictionary with the specified WQB or Jarzynski information depending on the mode.

    Args:
        info_dict (dict): System name -> list of collected values. For 'min', 'single' and 'avg' modes the list is
            [wqb, [(replica, wqb), ...]]; for 'jarzynski' it is [dG, sd, sem]; for 'all' the concatenation of both.
        mode (str): One of 'min', 'single', 'avg', 'jarzynski' or 'all'.
        sorting_column (str): Column used to sort the report.

    Returns:
        pandas.DataFrame: The report table.
    '''
    if mode == 'min':
        df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB': [wqb[0] for wqb in info_dict.values()]})
    elif mode == 'single':
        flat_dict = flatten_wqb_dict(info_dict)
        df = pd.DataFrame(flat_dict)
    elif mode == 'avg':
        df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB': [wqb[0] for wqb in info_dict.values()],
                           'Average': [_mean_sd(wqb[1])[0] for wqb in info_dict.values()],
                           'SD': [_mean_sd(wqb[1])[1] for wqb in info_dict.values()]})
    elif mode == 'jarzynski':
        df = pd.DataFrame({'System': list(info_dict.keys()),
                           'Jarzynski': [wqb[0] for wqb in info_dict.values()],
                           'Jarzynski_SD': [wqb[1] for wqb in info_dict.values()],
                           'Jarzynski_SEM': [wqb[2] for wqb in info_dict.values()]})
    elif mode == 'all':
        flat_dict = flatten_wqb_dict(info_dict)
        single_df = pd.DataFrame(flat_dict)
        wqb_df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB': [wqb[0] for wqb in info_dict.values()],
                               'Average': [_mean_sd(wqb[1])[0] for wqb in info_dict.values()],
                               'SD': [_mean_sd(wqb[1])[1] for wqb in info_dict.values()]})
        jarz_df = pd.DataFrame({'System': list(info_dict.keys()),
                                'Jarzynski': [wqb[-3] for wqb in info_dict.values()],
                                'Jarzynski_SD': [wqb[-2] for wqb in info_dict.values()],
                                'Jarzynski_SEM': [wqb[-1] for wqb in info_dict.values()]})
        df = pd.merge(wqb_df, jarz_df, on='System')
        df = pd.merge(df, single_df, on='System')
    else:
        raise ValueError(f'{mode} is not a valid report mode. Try with min, single, avg, jarzynski or all.')

    df.sort_values(sorting_column, inplace=True)
    return df


def find_wqb_min_index(work, segment=WQB_SEGMENT):
    """Find the first local minimum of a work profile, as in the original DUck analysis.

    The profile is split in segments of ``segment`` points. The minimum of each segment is accepted if it is
    also the minimum of a window of ``segment`` + 1 points centred on it; the first accepted minimum is used.
    Trailing points that do not fill a whole segment are not searched.

    Args:
        work (numpy.ndarray): Accumulated work profile.
        segment (int): Segment size in points (200 for the 5000-point AMBER profiles).

    Returns:
        int or None: Index of the minimum, or None if no local minimum was found.
    """
    half = segment // 2
    for i_segment in range(len(work) // segment):
        index_global = int(np.argmin(work[i_segment * segment:(i_segment + 1) * segment])) + i_segment * segment
        if index_global > half:
            window_start = index_global - half
        else:
            window_start = 0
        index_local = int(np.argmin(work[window_start:index_global + half + 1]))
        if index_global < half:
            # A minimum within the first half segment is accepted or rejected, the search stops either way
            return index_global if index_local == index_global else None
        if index_local == half:
            return index_global
    return None


def get_Wqb_value(file_duck_dat, mode='amber'):
    """Calculates the Wqb value from the DUck_work report file.

    Args:
        file_duck_dat (str): The path to the duck report file.
        mode (str, optional): Deprecated and ignored; the file format is detected automatically
            (see :func:`read_duck_dat`). Kept for backward compatibility.

    Returns a tupple with:
        float: The Wqb value calculated from the duck report (0 if no work minimum is found).
        numpy.ndarray: The duck report as an array with columns r0 (A), r (A), force and work.
        float: The minimum value of the Work column in the duck report, used in the Wqb calculation.
    """
    data = read_duck_dat(file_duck_dat)
    work = data[:, 3]
    wqb_min_index = find_wqb_min_index(work)
    if wqb_min_index is None:
        eprint(f'Warning: could not find minima in {file_duck_dat}')
        return (0, data, 0)
    wqb_min = work[wqb_min_index]
    wqb_max = np.max(work[wqb_min_index:])
    return wqb_max - wqb_min, data, wqb_min


def _collect_wqb(dat_files, plot=False):
    '''Compute the Wqb of each SMD report in `dat_files`; return (lowest wqb or 'Nan', [(name, wqb), ...]).'''
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    wqb_values = []
    for name, dat_file in dat_files:
        wqb, data, wqb_min = get_Wqb_value(dat_file)
        wqb_values.append((name, wqb))
        if plot:
            ax.plot(data[:, 0], data[:, 3] - wqb_min)
    if plot:
        ax.set_xlabel(r'HB Distance ($\AA$)')
        ax.set_ylabel(r'Work ($kcal · mol^{-1}$)')
        fig.savefig('wqb_plot.png')
        _close(fig)
    if len(wqb_values):
        wqb = min([x[1] for x in wqb_values])
    else:
        wqb = 'Nan'
    return wqb, wqb_values


def get_Wqb_value_Openmm_all(folder='duck_runs', pattern='smd_*.dat', plot=False):
    '''
    Obtain the Wqb for all replicas in a openduck run from an openMM simulation.

    Returns:
        tuple: (lowest Wqb, or 'Nan' if there are no replicas, [(dat_file, wqb), ...]) sorted by replica.
    '''
    dat_files = sorted(glob.glob(os.path.join(folder, pattern)), key=_natural_key)
    return _collect_wqb([(f, f) for f in dat_files], plot=plot)


def get_Wqb_value_AMBER_all(prefix='DUCK', file='duck.dat', plot=False):
    '''
    Obtain the Wqb for all replicas in dynamic undocking simulation in AMBER.

    All folders in the current directory starting with `prefix` (both DUCK_n and DUCK_325K_n) are considered.

    Returns:
        tuple: (lowest Wqb, or 'Nan' if there are no replicas, [(folder, wqb), ...]) sorted by replica.
    '''
    folders = sorted([fol for fol in os.listdir(os.getcwd()) if fol.startswith(prefix)], key=_natural_key)
    dat_files = [(fol, os.path.join(fol, file)) for fol in folders if os.path.isfile(os.path.join(fol, file))]
    return _collect_wqb(dat_files, plot=plot)


def get_Wqb_value_all(mode=None, plot=False):
    '''
    Obtain the Wqb of all replicas of the run in the current directory, including those of independent
    equilibrations (``EQ_<k>/``).

    Args:
        mode (str, optional): 'amber' (``DUCK_*/duck.dat``) or 'openmm' (``duck_runs/smd_*.dat``); detected if None.
        plot (bool): plot the work profiles to wqb_plot.png.

    Returns:
        tuple: (lowest Wqb, or 'Nan' if there are no replicas, [(replica, wqb), ...]). Replicas are named as in
        :func:`get_Wqb_value_AMBER_all` and :func:`get_Wqb_value_Openmm_all`, with the ``EQ_<k>/`` prefix if any.
    '''
    mode = mode or _detect_engine_layout()
    dat_files = []
    for root in find_equilibrations('.') or ['.']:
        if mode == 'amber':
            folders = sorted([fol for fol in os.listdir(root) if fol.startswith('DUCK')
                              and os.path.isfile(os.path.join(root, fol, 'duck.dat'))], key=_natural_key)
            dat_files += [(os.path.normpath(os.path.join(root, fol)), os.path.join(root, fol, 'duck.dat')) for fol in folders]
        else:
            files = sorted(glob.glob(os.path.join(root, 'duck_runs', 'smd_*.dat')), key=_natural_key)
            dat_files += [(os.path.normpath(f), f) for f in files]
    return _collect_wqb(dat_files, plot=plot)


# jarzynksi functions
_OPENMM_DAT = re.compile(r'smd_\d+_(\d+)\.dat$')
_AMBER_DIR_HOT = re.compile(r'^DUCK_(\d+)K_\d+$')
_AMBER_DIR = re.compile(r'^DUCK_\d+$')


def smd_temperature(dat_file):
    '''Return the simulation temperature (K) encoded in an SMD report path, or None if unknown.

    OpenMM reports are named smd_<cycle>_<T>.dat; AMBER reports live in DUCK_<cycle> (300 K) or
    DUCK_<T>K_<cycle> folders.
    '''
    match = _OPENMM_DAT.search(os.path.basename(dat_file))
    if match:
        return int(match.group(1))
    parent = os.path.basename(os.path.dirname(os.path.normpath(dat_file)))
    match = _AMBER_DIR_HOT.match(parent)
    if match:
        return int(match.group(1))
    if _AMBER_DIR.match(parent):
        return 300
    return None


def read_DUckdat(temp=300, pattern='DUCK*/duck.dat', work_col=3, CV_col=0):
    '''Read the work profiles of all SMD reports matching `pattern` into a DataFrame.

    Only replicas simulated at `temp` are kept (see :func:`smd_temperature`); this matters to calculate the
    Jarzynski free energy accurately. For paths that do not follow the DUck naming conventions the old
    heuristic is used ('325' in the path means 325 K). If `temp` is None all files are read.

    Args:
        temp (int): Temperature (K) of the replicas to read.
        pattern (str): Glob pattern of the SMD reports.
        work_col, CV_col: Deprecated and ignored; columns are found through :func:`read_duck_dat`.

    Returns:
        pandas.DataFrame: Work (kcal/mol) with one column per report and the target distance (rounded to
        4 decimals, as a string) as index.
    '''
    work_values, RC = {}, None
    for dat_file in sorted(glob.glob(pattern), key=_natural_key):
        if temp is not None:
            file_temp = smd_temperature(dat_file)
            if file_temp is None:  # unknown naming, keep the legacy heuristic
                file_temp = 325 if '325' in dat_file else 300
            if file_temp != temp:
                continue
        data = read_duck_dat(dat_file)
        file_rc = [str(round(float(x), 4)) for x in data[:, 0]]  # keep 4 decimals, otherwise amber steps overlap
        if RC is not None and len(file_rc) != len(RC):
            raise ValueError(f'{dat_file} has {len(file_rc)} rows but previous reports have {len(RC)}; '
                             'all replicas must have the same length.')
        RC = file_rc
        work_values[dat_file] = data[:, 3]
    return pd.DataFrame(work_values, index=RC if RC is not None else [])


def _default_threshold(n_rows, index_threshold):
    '''Resolve the minimum-search window: first half of the profile unless a valid explicit value is given.'''
    if index_threshold is None:
        return n_rows // 2
    if n_rows and index_threshold >= n_rows:
        eprint(f'Warning: step threshold {index_threshold} is not smaller than the number of points in the work '
               f'profiles ({n_rows}); using the first half ({n_rows // 2}) instead.')
        return n_rows // 2
    return index_threshold


def normalize_by_series(df, index_threshold=None):
    '''Shift every work profile (column) so that its minimum within the first `index_threshold` points is 0.

    Args:
        df (pandas.DataFrame): Work profiles, one column per replica.
        index_threshold (int, optional): Number of initial points searched for the minimum. Defaults to the
            first half of the profile (2500 points for AMBER-style reports).
    '''
    index_threshold = _default_threshold(len(df), index_threshold)
    return df - df.iloc[:index_threshold].min()


_FD_COLUMNS = ['expavg', 'sqrtMSE', 'MSE', 'Bj', 'v', 'av', 'ai', 'Wdis', 'avg', 'variance', 'FD', 'sqrtMSE_FD']


def get_expavg_FD_df(work_df, T=300, calculate_FD=True):
    '''Exponential (Jarzynski) average and fluctuation-dissipation estimates for each reaction coordinate step.

    Adapted from Luca's script. For each row (distance step) of `work_df`, with N replicas and B = 1/(kB T):

    * expavg = -ln(<exp(-B W)>) / B, the Jarzynski free energy;
    * Wdis = B var(W) / 2, the dissipated work, and FD = <W> - Wdis, the fluctuation-dissipation estimate;
    * Bj, v, MSE, sqrtMSE: bias, variance and mean squared error of the Jarzynski estimator, with the
      empirical exponents ai and av (both 1 when Wdis < 0.02 kcal/mol);
    * sqrtMSE_FD: error of the FD estimate.

    Args:
        work_df (pandas.DataFrame): Work values, one row per distance step and one column per replica.
        T (float): Temperature in K.
        calculate_FD (bool): Also compute the fluctuation-dissipation columns (they are NaN otherwise).

    Returns:
        pandas.DataFrame: Columns ``expavg, sqrtMSE, MSE, Bj, v, av, ai, Wdis, avg, variance, FD, sqrtMSE_FD``
        with the same index as `work_df`.
    '''
    B = 1 / (KB_KCAL * T)
    work = work_df.to_numpy(dtype=float)
    n_rows, N = work.shape
    result = {col: np.full(n_rows, np.nan) for col in _FD_COLUMNS}
    if n_rows == 0 or N == 0:
        return pd.DataFrame(result, index=work_df.index, columns=_FD_COLUMNS)

    average = work.mean(axis=1)
    variance = work.var(axis=1)
    result['avg'], result['variance'] = average, variance
    # Boltzmann average for the step, log-sum-exp avoids overflows of exp(-B W)
    result['expavg'] = -(logsumexp(-B * work, axis=1) - np.log(N)) / B

    if calculate_FD:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Dissipated work due to heat loss, Wdis = T*dS-dQ if Wdis >=0
            wdis = 0.5 * B * variance
            negligible = np.isnan(wdis) | (wdis < 0.02)  # for very small Wdis we can neglect it
            ai = np.where(negligible, 1.0, np.log(30 * wdis * B) / np.log(15 * (np.exp(2 * B * wdis) - 1)))
            av = np.where(negligible, 1.0, np.log(100 * wdis * B) / np.log(50 * (np.exp(2 * B * wdis) - 1)))
            bj = wdis / np.power(float(N), ai)
            v = variance / np.power(float(N), av)
            mse = bj ** 2 + v
            # Proper fluctuation dissipation (FD) using Kullback-leibler divergence
            mse_fd = np.where(wdis != 0, 2 * wdis / (B * N) + (2 * wdis ** 2) / (N - 1), 0.0)
        result.update({'Wdis': wdis, 'ai': ai, 'av': av, 'Bj': bj, 'v': v, 'MSE': mse, 'sqrtMSE': mse ** 0.5,
                       'FD': average - wdis, 'sqrtMSE_FD': mse_fd ** 0.5})
    return pd.DataFrame(result, index=work_df.index, columns=_FD_COLUMNS)


def plot_expavg_FD(raw_data_df, FD_df):
    '''Plot the normalized work of every replica together with the Jarzynski (green) and FD (red) profiles.'''
    import matplotlib.pyplot as plt
    RC = [float(x) for x in raw_data_df.index.values]
    fig, ax = plt.subplots()
    ax.plot(RC, raw_data_df.to_numpy(dtype=float), 'b', linewidth=0.5)
    # Plot expavg (Boltzmann avg) & FD (Fluctuation dissipation)
    ax.plot(RC, FD_df['expavg'].values, 'g', linewidth=4)
    ax.plot(RC, FD_df['FD'].values, 'r', linewidth=4)
    ax.set_ylabel('Free Energy (kcal/mol)')
    ax.set_xlabel('Distance (Å)')
    fig.savefig('wqb_plot_jarzynski.png')
    _close(fig)


def shapiro_test(work_df):
    '''Shapiro-Wilk test on the maximum normalized work of each replica; plots the histogram to wqb_hist.pdf.

    Returns:
        bool: True if normality cannot be rejected (p >= 0.05). False if it is rejected or if there are three
        or fewer replicas.
    '''
    import matplotlib.pyplot as plt
    # get the real maximum for the shapiro, as it will be the one used
    values = np.array([work_df[coll].max() for coll in work_df.columns])
    if len(values) <= 3:
        eprint('Warning: Not enough data to perform a shapiro test')
        return False
    statistic, p = shapiro(values)
    fig, ax = plt.subplots()
    y, x, _ = ax.hist(values, bins=len(values) // 2)
    ax.text(y=max(y) - 1, x=max(x) - 1.5, s='Shapiro p = %s' % round(p, 4))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Wqb (kcal/mol)')
    fig.savefig('wqb_hist.pdf')
    _close(fig)
    return bool(p >= 0.05)


def sample_jarz(norm_df, sample_size, temp, rng=None, groups=None):
    '''Jarzynski profile of one bootstrap sample: `sample_size` replicas drawn with replacement.

    Args:
        norm_df (pandas.DataFrame): Normalized work profiles, one column per replica.
        sample_size (int): Number of replicas drawn.
        temp (float): Temperature in K.
        rng (numpy.random.Generator, optional): Random generator; a freshly seeded one if None.
        groups (sequence, optional): equilibration of each column. With more than one, the equilibrations are
            resampled first and the replicas are drawn from the resampled ones (two-stage bootstrap).
    '''
    if rng is None:
        rng = np.random.Generator(np.random.PCG64())
    if groups is not None and len(set(groups)) > 1:
        indices, _ = cluster_bootstrap_indices(groups, sample_size, rng)
        new_df = norm_df.iloc[:, indices]
    else:
        new_df = norm_df.sample(n=sample_size, replace=True, axis='columns', random_state=rng)
    new_df.columns = [f'Sample_{x}' for x in range(sample_size)]
    jarz_df = get_expavg_FD_df(new_df, T=temp, calculate_FD=False)
    return jarz_df


def bootstrap_df(norm_data_list, sample_size=20, samples=20, plot=True, temps=[300, 325], seed=None, groups=None):
    """
    Perform bootstrap sampling of the normalized work dataframes to estimate the distribution of the Jarzynski free energy
    estimate. Each temperature is sampled separatedly and then merged when the jarzynski is already calculated.

    Args:
    norm_data_list: list of pandas.DataFrame, each dataframe has the normalized work values for each CV bin at a given temperature
    sample_size: int, the number of columns (work samples) to sample from each CV bin dataframe in each bootstrap iteration
    samples: int, the number of bootstrap iterations to perform for each temperature
    plot: bool, whether to plot the bootstrapped Jarzynski free energy distribution
    temps: list of ints, the temperatures at which to perform the bootstrapping
    seed: int or None, seed of the random generator (None for a non-reproducible run)
    groups: list (one per dataframe) of the equilibration of each column, or None. Runs with several independent
            equilibrations are resampled in two stages (see :func:`sample_jarz`).

    Returns:
    flat_df: pandas.DataFrame, a flat dataframe of bootstrapped Jarzynski free energy estimates and the corresponding CV value
    sample_dfs: list of pandas.DataFrame, a list of the bootstrapped samples for each temperature, containing the work values
                for each CV bin in each bootstrap iteration
    """
    rng = np.random.default_rng(seed)
    sample_dfs = []
    groups = groups if groups is not None else [None] * len(norm_data_list)
    for temp, norm_df, df_groups in zip(temps, norm_data_list, groups):
        if not len(norm_df):
            continue
        for i in range(samples):
            sample_dfs.append(sample_jarz(norm_df, sample_size, temp, rng=rng, groups=df_groups))
    if not sample_dfs:
        return pd.DataFrame({'CV': [], 'Jarzynski': []}), sample_dfs
    flat_df = pd.DataFrame({
        'CV': np.concatenate([sample.index.to_numpy(dtype=float) for sample in sample_dfs]),
        'Jarzynski': np.concatenate([sample['expavg'].to_numpy(dtype=float) for sample in sample_dfs]),
    })
    if plot:
        import matplotlib.pyplot as plt
        RC = [float(x) for x in sample_dfs[0].index]
        grouped = flat_df.groupby('CV', sort=False)['Jarzynski']
        avg_jarzs = grouped.mean().reindex(RC).to_numpy()
        std_jarzs = grouped.std(ddof=0).reindex(RC).to_numpy()
        fig, ax = plt.subplots()
        ax.errorbar(x=RC, y=avg_jarzs, yerr=std_jarzs, color='lightblue', zorder=0)
        ax.plot(RC, avg_jarzs, color='dodgerblue')
        ax.set_ylabel('Free Energy (kcal/mol)')
        ax.set_xlabel('Distance (Å)')
        fig.savefig('bootstraped_WQB_plot.png')
        _close(fig)
    return flat_df, sample_dfs


def get_stats_from_bootstrapping(flat_df, CVs):
    '''Mean, standard deviation and standard error of the bootstrapped Jarzynski free energy at each CV value.'''
    grouped = flat_df.groupby('CV', sort=False)['Jarzynski']
    CVs = [float(step) for step in CVs]
    return pd.DataFrame({'CV': CVs,
                         'AvgJarzynski': grouped.mean().reindex(CVs).to_numpy(),
                         'sd': grouped.std(ddof=0).reindex(CVs).to_numpy(),
                         'sem': grouped.sem(ddof=1).reindex(CVs).to_numpy()})


def _average_frames(frames):
    '''Element-wise average of the non-empty dataframes (aligned on the first one's labels).'''
    frames = [df for df in frames if len(df)]
    if not frames:
        return pd.DataFrame()
    reference = frames[0]
    total = reference.astype(float)
    for df in frames[1:]:
        total = total + df.loc[reference.index, reference.columns].astype(float)
    return total / len(frames)


def average_dataframes(dataf1, dataf2):
    '''Return the average dataframe of two dataframes. Used to average the fluctuation-dissipation dataframes of the
    two temperatures; if one of them is empty the other is returned.'''
    if not len(dataf1) or not len(dataf2):
        eprint('Warning: Not enough data to merge different temperatures')
    if not len(dataf1):
        return dataf2
    elif not len(dataf2):
        return dataf1
    return _average_frames([dataf1, dataf2])


def get_real_jarzynski_from_bootstrapping(sample_dfs, save=None, splitting_point=None):
    '''dG_QB of each bootstrap sample and their statistics.

    The calculated dG_QB comes from the maximum after the minimum of the Jarzynski profile, not from the end point.
    The minimum is searched within the first `splitting_point` points (first half of the profile by default).

    Returns:
        tuple: (mean, standard deviation, standard error of the mean, list of per-sample dG_QB). If `save` is given
        the first three are written tab-separated to that file.
    '''
    max_values = []
    for sample_df in sample_dfs:
        expavg = sample_df['expavg'].to_numpy(dtype=float)
        split = _default_threshold(len(expavg), splitting_point)
        min_index = int(np.nanargmin(expavg[:split]))
        max_values.append(float(np.nanmax(expavg[min_index:])))
    avg = np.mean(max_values)
    sd = np.std(max_values)
    sem_v = sem(max_values)
    if save:
        with open(save, 'w') as f:
            f.write('\t'.join([str(x) for x in [avg, sd, sem_v]]))
    return avg, sd, sem_v, max_values


def _detect_engine_layout():
    '''Guess the SMD output layout of the current directory: 'openmm' (duck_runs/smd_*.dat) or 'amber'.'''
    roots = find_equilibrations('.') or ['.']
    return 'openmm' if any(glob.glob(os.path.join(root, 'duck_runs', 'smd_*.dat')) for root in roots) else 'amber'


def do_jarzynski_analysis(temperatures=[300, 325], index_threshold=None, sample_size=20, samples=20, plot=True,
                          mode='amber', seed=None):
    """
    Performs the Jarzynski analysis on a set of SMD data at different temperatures. To assess the significance of the values obtained, bootstrapping is performed.
    The replicas of independent equilibrations (``EQ_<k>/``) are pooled and bootstrapped in two stages.

    Args:
        temperatures (list, optional): A list of temperatures (in Kelvin) at which the simulations were performed. Defaults to [300,325].
        index_threshold (int, optional): Number of initial points where the work minimum is searched (normalization and dG_QB).
            Defaults to the first half of the profiles (2500 points for 5000-point reports).
        sample_size (int, optional): The size of each bootstrap sample. Defaults to 20.
        samples (int, optional): The number of bootstrap samples to take. Defaults to 20.
        plot (bool, optional): Whether to generate plots or not. Defaults to True.
        mode (str, optional): Layout of the SMD outputs: 'amber' (DUCK_*/duck.dat) or 'openmm' (duck_runs/smd_*_T.dat).
            None detects it from the current directory. Defaults to 'amber'.
        seed (int, optional): Seed for the bootstrapping. Defaults to None (non-reproducible).

    Returns:
        tuple: A tuple containing the exponential average, standard deviation, and standard error of the mean of the Jarzynski free energy.
    """
    if mode is None:
        mode = _detect_engine_layout()
    norm_datas, FD_datas, groups = [], [], []
    prefix = f'{EQUILIBRATION_PREFIX}_[0-9]*/' if find_equilibrations('.') else ''
    for T in temperatures:
        if mode == 'amber':
            pattern = prefix + 'DUCK_*/duck.dat'
        elif mode == 'openmm':
            pattern = prefix + f'duck_runs/smd_*_{T}.dat'
        else:
            raise ValueError(f'{mode} is not a valid SMD output layout; use amber or openmm.')
        WQB_df = read_DUckdat(pattern=pattern, temp=T)
        norm_df = normalize_by_series(WQB_df, index_threshold=index_threshold)
        FD_datas.append(get_expavg_FD_df(norm_df, T=T))
        norm_datas.append(norm_df)
        groups.append([equilibration_of(column) for column in norm_df.columns] if prefix else None)
    if not any(len(df.columns) for df in norm_datas):
        raise ValueError(f'No SMD work reports found in {os.getcwd()} for the {mode} layout.')
    norm_df = pd.concat(norm_datas, axis=1)
    CVs = list(norm_df.index)
    # check normality of data
    shapiro_test(norm_df)

    # get data for fluctuation-dissipation
    if len(FD_datas) == 2:
        FD_df = average_dataframes(*FD_datas)
    else:
        FD_df = _average_frames(FD_datas)
    if plot:
        plot_expavg_FD(norm_df, FD_df)

    # obtained bootstrapped data on jarzynski
    flat_bootstrapped_df, sampled_dfs = bootstrap_df(norm_datas, sample_size=sample_size, samples=samples, plot=plot,
                                                     temps=temperatures, seed=seed, groups=groups)
    stats_df = get_stats_from_bootstrapping(flat_bootstrapped_df, CVs)
    avg, sd, sem_v, max_stats = get_real_jarzynski_from_bootstrapping(sampled_dfs, save='jarz_sd_sem.tbl',
                                                                      splitting_point=index_threshold)

    # save data in csv
    FD_df.to_csv('fluctuation_dissipation.csv')
    stats_df.to_csv('bootstrapped_stats.csv')
    return avg, sd, sem_v


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect data from duck to report. Prefer `openduck report`.')
    parser.add_argument('-p', '--pattern', default='.', type=str, help='Bash wildcard pattern to find folders with duck data')
    parser.add_argument('-m', '--mode', type=str, default='min', choices=('min', 'single', 'avg', 'jarzynski', 'all'), help='Mode to compile the report [min | single | avg | jarzynski | all]')
    parser.add_argument('-o', '--output', default=sys.stdout, help='Output file, default is printing report to stdout.')
    parser.add_argument('-of', '--output_format', default='tbl', choices=('csv', 'sdf', 'tbl'), help='Output format, [csv | sdf | tbl].')
    parser.add_argument('--plot', default=False, action='store_true', help='Plot work or energy values to file.')
    args = parser.parse_args()

    wqb_info = {}
    currdir = os.getcwd()
    for folder in glob.glob(args.pattern):
        wqb_info.setdefault(folder, [])
        os.chdir(folder)
        layout = _detect_engine_layout()
        if args.mode in ('min', 'single', 'avg', 'all'):
            if layout == 'openmm':
                wqb_info[folder].extend(get_Wqb_value_Openmm_all(plot=args.plot))
            else:
                wqb_info[folder].extend(get_Wqb_value_AMBER_all(plot=args.plot))
        if args.mode in ('jarzynski', 'all'):
            wqb_info[folder].extend(do_jarzynski_analysis(plot=args.plot, mode=layout))
        os.chdir(currdir)

    df = build_report_df(wqb_info, mode=args.mode)
    if args.output_format == 'csv':
        df.to_csv(args.output, index=False)
    elif args.output_format == 'tbl':
        df.to_csv(args.output, index=False, sep='\t')
    elif args.output_format == 'sdf':
        from rdkit import Chem
        mols = get_mols_and_format(df, mode=args.mode)
        with Chem.SDWriter(args.output) as w:
            for mol in mols:
                w.write(mol)
