#!/usr/bin/env python
import glob
import os
import numpy as np
import pandas as pd
import sys
from rdkit import Chem
from scipy.stats import shapiro, sem
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, exp
import pickle
import time

#base functions
def save_pickle(file, object):
    with open(file, 'wb') as fh:
        pickle.dump(object, fh)
def read_pickle(file):
    with open(file, 'r') as fh:
        obj = pickle.load(fh)
    return obj
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

#wqb report functions
def get_mols_and_format(data_df, mode='min'):
    '''Writeout an rdkit mol object with the specified WQB or Jarzynski information depending on the mode.
    It uses the ligand.pdb file from each DUck directory.'''
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
            mol.SetProp('WQB', str(row['WQB']) )
        elif mode == 'single':
            data = list(row.values)
            data.pop(0) # pop system
            mol.SetProp('All WQB' ,','.join([str(x) for x in data]))
        elif mode == 'avg':
            mol.SetProp('WQB', str(row['WQB']) )
            mol.SetProp('WQB AVG', str(row['Average']) )
            mol.SetProp('WQB SD', str(row['SD']) )
        elif mode == 'jarzynski':
            mol.SetProp('Jarzynski', str(row['Jarzynski']) )
            mol.SetProp('Jarzynski_SD', str(row['Jarzynski_SD']) )
            mol.SetProp('Jarzynski_SEM', str(row['Jarzynski_SEM']) )
        elif mode == 'all':
            mol.SetProp('WQB', str(row['WQB']) )
            mol.SetProp('WQB AVG', str(row['Average']) )
            mol.SetProp('WQB SD', str(row['SD']) )
            mol.SetProp('Jarzynski', str(row['Jarzynski']) )
            mol.SetProp('Jarzynski_SD', str(row['Jarzynski_SD']) )
            mol.SetProp('Jarzynski_SEM', str(row['Jarzynski_SEM']) )
            data = list(row.values)
            mol.SetProp('All WQB' ,','.join([str(x) for x in data[7:]]))
        mols.append(mol)
    return mols

def flatten_wqb_dict(info_dict):
    """
    Flatten a dictionary of Wqb values into a more easily digestible format.

    Parameters:
    info_dict (dict): A dictionary with keys as system names and values as tuples containing
    the system name and a list of tuples, where each tuple contains a residue name and its Wqb value.

    Returns:
    dict: A dictionary where the keys are 'System' and the residue names, and the values are lists
    of the corresponding Wqb values for each system.
    """
    values = list(info_dict.values())
    k = ['System', *[x[0] for x in values[0][1]]]
    v = [list(info_dict.keys())]
    v.extend([[] for i in range(len(values[0][1]))])
    for i in range(len(values)):
        for j in range(len(values[0][1])):
            if len(k)-1 == len(values[i][1]):
                v[j+1].append(values[i][1][j][1])
            else:
                v[j+1].append('Nan')
    return {ks: vs for ks, vs in zip(k, v)}

def build_report_df(info_dict, mode='min', sorting_column='System'):
    import re
    '''Generate a report DataFrame from the dictionary with the specified WQB or Jarzynski information depending on the mode'''
    if mode == 'min':
        df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB':[wqb[0] for wqb in info_dict.values()]})
    elif mode == 'single':
        flat_dict = flatten_wqb_dict(info_dict)
        df = pd.DataFrame(flat_dict)
    elif mode == 'avg':
        df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB':[wqb[0] for wqb in info_dict.values()],
                            'Average': [np.mean([x[1] for x in wqb[1]]) for wqb in info_dict.values()],
                            'SD': [np.std([x[1] for x in wqb[1]]) for wqb in info_dict.values()]})
    elif mode == 'jarzynski':
       df = pd.DataFrame({'System': list(info_dict.keys()),
                            'Jarzynski':[wqb[0] for wqb in info_dict.values()],
                            'Jarzynski_SD':[wqb[1] for wqb in info_dict.values()],
                            'Jarzynski_SEM':[wqb[2] for wqb in info_dict.values()],}) 
    elif mode == 'all':
        flat_dict = flatten_wqb_dict(info_dict)
        single_df = pd.DataFrame(flat_dict)
        wqb_df = pd.DataFrame({'System': list(info_dict.keys()), 'WQB':[wqb[0] for wqb in info_dict.values()],
                            'Average': [np.mean([x[1] for x in wqb[1]]) for wqb in info_dict.values()],
                            'SD': [np.std([x[1] for x in wqb[1]]) for wqb in info_dict.values()]})
        jarz_df = pd.DataFrame({'System': list(info_dict.keys()),
                            'Jarzynski':[wqb[-3] for wqb in info_dict.values()],
                            'Jarzynski_SD':[wqb[-2] for wqb in info_dict.values()],
                            'Jarzynski_SEM':[wqb[-1] for wqb in info_dict.values()],}) 
        df = pd.merge(wqb_df, jarz_df, on='System')
        df = pd.merge(df, single_df, on='System')
    else:
        raise ValueError(f'{mode} is not a valid report mode. Try with min, all or avg.')
    
    df.sort_values(sorting_column, inplace=True)
    # Sorts better when there is numbers, fails when there isn't
    #df.sort_values(sorting_column, inplace=True, key=lambda col: col.map(lambda x: int(re.findall(r'\d+', x)[0])))
    return df

#from maciej
def get_Wqb_value(file_duck_dat, mode='amber'):
    """Calculates the Wqb value from the DUck_work report file.

    Args:
        file_duck_dat (str): The path to the duck report file.
        mode (str, optional): The format of the duck report file. Defaults to 'amber'.

    Returns a tupple with:
        float: The Wqb value calculated from the duck report.
        numpy.ndarray: The entire contents of the duck report file as a numpy array.
        float: The minimum value of the Work column in the duck report, used in the Wqb calculation.
    """
    f = open(file_duck_dat,'r')
    data = []
    for line in f:
        a = line.split()
        if mode == 'amber':
            if len(a) != 4:
                eprint(f'{file_duck_dat} has the wrong format for {mode} duck report' )
                exit(1)
            else:
                data.append([float(a[0]), float(a[1]), float(a[2]), float(a[3])])
        elif mode == 'openmm':
            if len(a) != 10:
                eprint(f'{file_duck_dat} has the wrong format for {mode} duck report' )
                exit(1)
            else:
                data.append([float(a[1]), float(a[3]), float(a[5]), float(a[8])])
    f.close()
    if len(data) == 0:
        raise ValueError(f'{file_duck_dat} is empty.')
    data = np.array(data)
    Work = data[:,3]
    #split it into segments of 200 points 
    num_segments = int(len(data)/200) 
    #alayze each segment to see if minimum in the segment is the local minimum
    #local minimum is the point with the lowest value of 200 neighbouring points
    #first local minumum is miminum used later to duck analysis
    Wqb_min_index = False
    for segment in range(num_segments):
        #detecting minium inthe segment
        sub_data = data[segment * 200 : (segment + 1) * 200]
        sub_Work = sub_data[:,3] 
        index_local = np.argmin(sub_Work)
        #segment of 200 points arround detected minimum
        index_global = index_local + segment * 200
        if index_global > 100:
            sub2_data = data[index_global - 100 : index_global + 101]
        else:
            sub2_data = data[0 : index_global + 101]
        sub2_Work = sub2_data[:,3]
        index_local2 = np.argmin(sub2_Work)
        if index_global < 100:
            if index_local2 == index_global:
                
                Wqb_min_index = index_global
            break
        else:
            if index_local2 == 100:
                Wqb_min_index = index_global
                break
    if not Wqb_min_index:
        eprint(f'Warning: could not find minima in {file_duck_dat}')
        return (0, data, 0)
    Wqb_min = Work[Wqb_min_index]
    sub_max_data = data[Wqb_min_index:]
    sub_max_Work = sub_max_data[:,3]
    Wqb_max_index = np.argmax(sub_max_Work)
    Wqb_max_index_global = Wqb_max_index + Wqb_min_index
    
    Wqb_max = max(sub_max_Work)
    
    Wqb_value = Wqb_max - Wqb_min
    
    return Wqb_value, data, Wqb_min

def get_Wqb_value_Openmm_all(folder='duck_runs', pattern='smd_*.dat', plot=False):
    '''
    Obtain the Wqb for all replicas in a openduck run from an openMM simulation.
    '''
    wqb_values = []
    if plot: fig, ax = plt.subplots()        
    for dat_file in glob.glob(os.path.join(folder, pattern)):
        wqb_data = get_Wqb_value(dat_file, mode = 'openmm')
        wqb_values.append((dat_file, wqb_data[0]))
        if plot: ax.plot(wqb_data[1][:,0], wqb_data[1][:,3]-wqb_data[2])
    if plot:
        ax.set_xlabel('HB Distance ($\AA$)')
        ax.set_ylabel('Work ($kcal · mol^{-1}$)')
        fig.savefig('wqb_plot.png')
        plt.close(fig)
    if len(wqb_values):
        wqb = min([x[1] for x in wqb_values])
    else:
        wqb = 'Nan'
    return wqb, wqb_values

def get_Wqb_value_AMBER_all(prefix = 'DUCK', file = 'duck.dat', plot=False):
    '''
    Obtain the Wqb for all replicas in dynamic undocking simulation in AMBER.
    '''
    #adapted from maciej script
    folder = []
    for fol in os.listdir(os.getcwd()):
        if fol.startswith(prefix):
            folder.append(fol)
    if plot: fig, ax = plt.subplots()        
    wqb_values = []
    for fol in folder:
        if os.path.isfile(os.path.join(fol,file)):
            wqb_data = get_Wqb_value(os.path.join(fol,file))
            wqb_values.append((fol, wqb_data[0]))
            if plot: ax.plot(wqb_data[1][:,0], wqb_data[1][:,3]-wqb_data[2])
    if plot:
        ax.set_xlabel('HB Distance ($\AA$)')
        ax.set_ylabel('Work ($kcal · mol^{-1}$)')
        fig.savefig('wqb_plot.png')
        plt.close(fig)
    if len(wqb_values):
        wqb = min([x[1] for x in wqb_values])
    else:
        wqb = 'Nan'
    return wqb, wqb_values

#jarzynksi functions
def read_DUckdat(temp = 300, pattern='DUCK*/duck.dat', work_col=3, CV_col=0):
    # read SMD reports from amber (4th line is the work)
    # If temp not None, it only takes the specified temperature works
    # its important in order to accurately calculate the jarzynski dG

    work_values, RC = {}, []
    for dat_file in glob.glob(pattern):
        # This check is because DUCK folders in amber duck version do not specify the temperature at 300K
        if temp == 300 and '325' in dat_file:
            continue
        elif temp == 325 and '325' not in dat_file:
            continue
        with open(dat_file) as fh:
            work_values[dat_file] = []
            RC = []
            for line in fh:
                line = line.split()
                work_values[dat_file].append(float(line[work_col]))
                RC.append(str(round(float(line[CV_col]),4))) # important to keep at least to 4 decimals. Otherwise amber inputs overlap steps
            if len(RC) == 0:
                raise ValueError(f'{dat_file} is empty, please check what is happening.')
    return pd.DataFrame(work_values, index = RC)

def normalize_by_series(df, index_threshold = 2500):
    # Normalize the work values to have the minima at 0 kcal/mol
    # Find the minima in the first half of steps (i.e 2500).
    # Should be tweaked but its the normal behaviour.
    import copy
    norm_df = copy.deepcopy(df)
    for col in norm_df.columns:
        norm_df[col] = norm_df[col]-min(norm_df[col][:index_threshold])
    return norm_df

def get_expavg_FD_df(work_df, T=300, calculate_FD=True):
    #Calcular Jar // adapted from Luca's script
    kB=0.0019872  #Units: kcal/mol/K
    B=1/(kB*T)
    final_df = pd.DataFrame(columns=['expavg', 'sqrtMSE', 'MSE', 'Bj', 'v', 'av', 'ai', 'Wdis', 'avg', 'variance', 'FD', 'sqrtMSE_FD'],
                            index = work_df.index)
    t_work = work_df.T
    expBwork  = t_work.applymap(lambda x : exp(x*(-B)))
    N = len(work_df.columns) # Should be 5000 for DUck

    for rc, work_step in work_df.iterrows():
        #initialize variables to prevent errors in saving empty values
        expavg, sqrtMSE, MSE, Bj, v, av, ai, Wdis, average, variance, FD, sqrtMSE_FD = (np.NAN for x in range(12))
        
        #Normal statistics
        average  = np.mean(work_step.array)
        variance = np.var(work_step)

        #Boltzman average for the step
        expavg   = (-log(np.mean(expBwork.loc[:,rc])))/B
        if calculate_FD: # I have added the option to skip it, as it was giving some overflow float problems with very big dissipations during sampling
            # Other parameter for the fluctuation dissipation formula (applied to the Boltzmann avg)
            # Dissipated work due to heat loss, Wdis = T*dS-dQ if Wdis >=0
            Wdis = 0.5*B*variance
            if np.isnan(Wdis) or Wdis < 0.02: # for very small Wdis we can neglect it
                ai, av = 1, 1
            else:
                ai = float(log(30*Wdis*B))/float(log(15*(exp(2*B*Wdis)-1)))
                av=float(log(100*Wdis*B))/float(log(50*(exp(2*B*Wdis)-1)))
            try:
                Bj = Wdis/N**ai
                v = variance/N**av
                MSE = Bj**2+v
                sqrtMSE = MSE**0.5
                # Proper fluctuation dissipation (FD) using Kullback-leibler divergence
                FD = average-Wdis
                if Wdis: MSE_FD = 2*Wdis/(B*N)+(2*Wdis**2)/(N-1)
                else: MSE_FD=0
                sqrtMSE_FD = MSE_FD**0.5
            except:
                pass



        # write out on the final_df
        final_df.loc[rc] = [expavg, sqrtMSE, MSE, Bj,v, av, ai, Wdis, average, variance, FD, sqrtMSE_FD]
    return final_df

def plot_expavg_FD(raw_data_df, FD_df):
    RC = [float(x) for x in raw_data_df.index.values]
    fig, ax = plt.subplots()
    # Plot raw Wqb
    for i, duck_file in raw_data_df.T.iterrows():
        ax.plot(RC,duck_file.values, 'b', linewidth=0.5)

    # Plot expavg (Boltzmann avg) & FD (Fluctuation dissipation)
    #RC = [float(x) for x in RC]	
    ax.plot(RC, FD_df['expavg'].values, 'g', linewidth=4)
    ax.plot(RC, FD_df['FD'].values,'r', linewidth=4)
    #ax.set_xticks(np.array([2.5,3.0,3.5,4.0,4.5,5.0]))
    #ax.set_xticks(np.append(RC[::1000], RC[-1]))
    #print(RC)

    ax.set_ylabel('Free Energy (kcal/mol)')
    ax.set_xlabel('Distance (\u212B)')
    fig.savefig('wqb_plot_jarzynski.png') 

def shapiro_test(work_df):
    #Perform a shapiro-wilk test to assess normality of the work values
    fig, ax = plt.subplots()
    #values = work_df.tail(1).values[0]
    values = np.array([work_df[coll].max() for coll in work_df.columns]) # get the real maximum for the shapiro, as it will be the one used
    if len(values) <= 3:
        eprint('Warning: Not enough data to perform a shapiro test')
        return False
    statistic, p = shapiro(values)
    y, x, _ = ax.hist(values, bins = len(values)//2)
    ax.text(y = max(y)-1, x = max(x)-1.5, s ='Shapiro p = %s'%round(p,4))
    sns.despine()
    ax.set_xlabel('Wqb (kcal/mol)')
    fig.savefig('wqb_hist.pdf')
    if p >= 0.05:
        return True
    else:
        return False

def sample_jarz(norm_df, sample_size, temp):
    new_df = norm_df.sample(n=sample_size, replace=True,axis='columns', random_state=np.random.Generator(np.random.PCG64()))
    new_df.columns = [f'Sample_{x}'for x in range(sample_size)]
    jarz_df = get_expavg_FD_df(new_df, T=temp, calculate_FD=False)
    return(jarz_df)

def bootstrap_df(norm_data_list, sample_size = 20, samples=20, plot=True, temps=[300, 325]):
    """
    Perform bootstrap sampling of the normalized work dataframes to estimate the distribution of the Jarzynski free energy
    estimate. Each temperature is sampled separatedly and then merged when the jarzynski is already calculated.

    Args:
    norm_data_list: list of pandas.DataFrame, each dataframe has the normalized work values for each CV bin at a given temperature
    sample_size: int, the number of columns (work samples) to sample from each CV bin dataframe in each bootstrap iteration
    samples: int, the number of bootstrap iterations to perform for each temperature
    plot: bool, whether to plot the bootstrapped Jarzynski free energy distribution
    temps: list of ints, the temperatures at which to perform the bootstrapping

    Returns:
    flat_df: pandas.DataFrame, a flat dataframe of bootstrapped Jarzynski free energy estimates and the corresponding CV value 
    sample_dfs: list of pandas.DataFrame, a list of the bootstrapped samples for each temperature, containing the work values 
                for each CV bin in each bootstrap iteration
"""
    import multiprocessing as mp
    if mp.cpu_count() >= samples:
        pool = mp.Pool(samples)
    else:
        pool = mp.Pool(mp.cpu_count())
    jobs = []
    for temp, norm_df in zip(temps, norm_data_list):
        if not len(norm_df): continue
        for i in range(samples):
            jobs.append(pool.apply_async(sample_jarz, args=(norm_df, sample_size, temp)))
    sample_dfs = [job.get() for job in jobs]
    flat_dict = {'CV':[],'Jarzynski':[]}
    for sample in sample_dfs:
        for i,row in sample.iterrows():
            flat_dict['CV'].append(float(i))
            flat_dict['Jarzynski'].append(row['expavg'])
    flat_df = pd.DataFrame(flat_dict)
    if plot:
        fig, ax = plt.subplots()
        RC = [float(x) for x in sample_dfs[0].index]
        avg_jarzs = [np.mean(list(flat_df[flat_df['CV'] == x]['Jarzynski'])) for x in RC]
        std_jarzs = [np.std(list(flat_df[flat_df['CV'] == x]['Jarzynski'])) for x in RC]
        ax.errorbar(x=RC, y=avg_jarzs,yerr=std_jarzs, color='lightblue', zorder=0)
        ax.plot(RC,avg_jarzs, color='dodgerblue')
        #ax.errorbar()
        ax.set_ylabel('Free Energy (kcal/mol)')
        ax.set_xlabel('Distance (\u212B)')
        fig.savefig('bootstraped_WQB_plot.png')
    return flat_df, sample_dfs

def get_stats_from_bootstrapping(flat_df, CVs):
    stats_dict = {'CV':[],'AvgJarzynski':[], 'sd':[], 'sem':[]}
    for step in CVs:
        sub_df = flat_df[flat_df['CV'] == float(step)]
        values = np.array(sub_df['Jarzynski'])
        stats_dict['CV'].append(float(step))
        stats_dict['AvgJarzynski'].append(np.mean(values))
        stats_dict['sd'].append(np.std(values))
        stats_dict['sem'].append(sem(values))
    stats_df = pd.DataFrame(stats_dict)
    return stats_df

def average_dataframes(dataf1, dataf2):
    '''Return the average dataframe of two dataframes. Used as the average dataframe for the Fluctuation dissipation dataframe at the two different'''
    if not len(dataf1) or not len(dataf2):
        eprint('Warning: Not enough data to merge different temperatures')
    if not len(dataf1):
        return dataf2
    elif not len(dataf2):
        return dataf1
    new_df = pd.DataFrame(index=dataf1.index, columns=dataf1.columns)
    for i,r in dataf1.iterrows():
        for col,v300 in r.items():
            v325 = dataf2.loc[i,col]
            new_df.at[i,col] = (v300+v325)/2
    return new_df

def get_real_jarzynski_from_bootstrapping(sample_dfs, save=None, splitting_point=2500):
    # real calculated wqb should come from the maximums and not from the end points
    # as such, the reported values come from the following
    max_values = [max(list(sample_df['expavg'])[list(sample_df['expavg'])[:splitting_point].index(min(list(list(sample_df['expavg'])[:splitting_point]))):])
                for sample_df in sample_dfs]
    avg = np.mean(max_values)
    sd = np.std(max_values)
    sem_v = sem(max_values)
    if save:
        with open(save, 'w') as f:
            f.write('\t'.join([str(x) for x in [avg,sd,sem_v]]))
    return avg,sd,sem_v, max_values

def do_jarzynski_analysis(temperatures = [300,325], index_threshold = 2500, sample_size = 20, samples=20, plot=True, mode='amber'):
    """
    Performs the Jarzynski analysis on a set of SMD data at different temperatures. To assess the significance of the values obtained, bootstrapping is performed.

    Args:
        temperatures (list, optional): A list of temperatures (in Kelvin) at which the simulations were performed. Defaults to [300,325].
        index_threshold (int, optional): The index at which the work data is truncated. Defaults to 2500.
        sample_size (int, optional): The size of each bootstrap sample. Defaults to 20.
        samples (int, optional): The number of bootstrap samples to take. Defaults to 20.
        plot (bool, optional): Whether to generate plots or not. Defaults to True.
        mode (str, optional): The simulation software used (`amber` or `openmm`). Defaults to 'amber'.

    Returns:
        tuple: A tuple containing the exponential average, standard deviation, and standard error of the mean of the Jarzynski free energy.
    """
    norm_datas, FD_datas = [],[]
    for T in temperatures:
        if mode == 'amber': pattern, work_col, CV_col = f'DUCK_*/duck.dat', 3, 0 # first col is HB distance and fourth is Work
        elif mode == 'openmm': pattern, work_col, CV_col = f'duck_runs/smd_*_{str(T)}.dat', 8, 1 # second column is HB distance and ninth is Work
        WQB_df = read_DUckdat(pattern=pattern, temp=T, work_col=work_col, CV_col=CV_col)
        norm_df = normalize_by_series(WQB_df, index_threshold=index_threshold)
        FD_df = get_expavg_FD_df(norm_df,T=T)
        norm_datas.append(norm_df)
        FD_datas.append(FD_df)
    norm_df = pd.concat(norm_datas, axis=1)
    CVs = list(norm_df.index)
    # check normality of data
    normality = shapiro_test(norm_df)

    # get data for fluctuation-dissipation
    FD_df = average_dataframes(*FD_datas)
    if plot: plot_expavg_FD(norm_df, FD_df)

    # obtained bootstrapped data on jarzynski
    flat_bootstrapped_df, sampled_dfs = bootstrap_df(norm_datas, sample_size = sample_size, samples=samples, plot=plot, temps=temperatures)
    
    #save_pickle('resampling.pickle',(flat_bootstrapped_df, sampled_dfs)) # migh remove pickle saving

    stats_df = get_stats_from_bootstrapping(flat_bootstrapped_df, CVs)
    avg,sd,sem_v, max_stats = get_real_jarzynski_from_bootstrapping(sampled_dfs, save='jarz_sd_sem.tbl')

    #save data in csv
    FD_df.to_csv('fluctuation_dissipation.csv')
    stats_df.to_csv('bootstrapped_stats.csv')
    return avg, sd, sem_v

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect data from duck to report')
    parser.add_argument('-p', '--pattern', default='.', type=str, help='Bash wildcard pattern to find folders with duck data')
    parser.add_argument('-m', '--mode', type=str, default='min', choices=('min', 'single', 'avg', 'jarzynski', 'all'), help='Mode to compile the report [min | single | avg | jarzynski | all]')
    parser.add_argument('-o', '--output', default='stdout', help = 'Output file, default is printing report to stdout.')
    parser.add_argument('-of', '--output_format', default='tbl', type=str, help='Output format, [csv | sdf | tbl].')
    parser.add_argument('--plot', default=False, action='store_true', help='Plot work or energy values to file.')

    args = parser.parse_args()

    start_time = time.time()

    if args.output == 'stdout':
        args.output = sys.stdout
        
    #iterate_folders
    folders = glob.glob(args.pattern)
    wqb_info = {}
    for folder in folders:
        wqb_info.setdefault(folder, [])
        #calculate_wqb
        currdir = os.getcwd()
        os.chdir(folder)

        if args.mode in ('min', 'single', 'avg', 'all'):
            wqb = get_Wqb_value_AMBER_all(plot=args.plot)
            wqb_info[folder].extend(wqb)
        if args.mode == 'jarzynski' or args.mode == 'all':
            expavg, sd, sem_v = do_jarzynski_analysis(plot=args.plot)
            wqb_info[folder].extend([expavg, sd, sem_v])

        os.chdir(currdir)

    df = build_report_df(wqb_info, mode=args.mode)
    if args.output_format == 'csv':
        df.to_csv(args.output, index=False)
    elif args.output_format == 'tbl':
        df.to_csv(args.output, index=False, sep='\t')
    elif args.output_format == 'sdf':
        mols = get_mols_and_format(df, mode=args.mode)
        with Chem.SDWriter(args.output) as w:
            [w.write(mol) for mol in mols]