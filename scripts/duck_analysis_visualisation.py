import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from duck.utils.vmd_template import get_chunk_residue, vmd_template
from duck.utils.analysis_and_report import read_duck_dat, find_wqb_min_index

def get_Wqb_value(file_duck_dat):
    """
    WQB of one SMD report (any format supported by duck.utils.analysis_and_report.read_duck_dat).

    Unlike the library version, the first row is skipped and, if no local work minimum is found, the global
    minimum of the trace is used instead of returning 0.
    :param file_duck_dat:
    :return: (wqb, data, wqb_min), data columns being r0 (A), r (A), force and work
    """
    data = read_duck_dat(file_duck_dat)[1:]
    Work = data[:, 3]
    Wqb_min_index = find_wqb_min_index(Work)
    # For some of my traces, the local minimum doesn't work. IN that case, try defaulting to the global minimum.
    if Wqb_min_index is None:
        print('Cannot find minimum Wqb index for file {} \n Setting it to the minimium work observed in the trace'.format(file_duck_dat))
        Wqb_min_index = np.argmin(Work)

    Wqb_min = Work[Wqb_min_index]
    Wqb_max = max(Work[Wqb_min_index:])

    Wqb_value = Wqb_max - Wqb_min
    return (Wqb_value, data, Wqb_min)

def make_all_data_dict(dir_paths, num_runs=40):
    dat_dict = {}
    to_sim = list(set([(x).name.split("_smd")[0] for x in all_runs]))
    for i, dp in enumerate(dir_paths):
        print(to_sim[i])
        try:
            all_data = [get_Wqb_value(p) for p in dp]
            dat_dict[to_sim[i]] = [(p, i) for p, i in zip(dp, all_data)]
        except IndexError:
            continue
        if len(all_data) < num_runs:
            continue
        
    return dat_dict

def make_wqb_images(dat_dict, save_dir):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir()
        
    for interaction, run in dat_dict.items():
        all_data=[run[i][1] for i in range(len(run))]
        plt.figure()
        for it, d in enumerate(all_data):
            dists = 1*d[1][:, 0]
            Work = d[1][:,3]-d[2]
            plt.plot(dists, Work)
            plt.text(x=dists[-1], y=Work[-1], s=str(round(d[0], 2)) + ' ' + str(it))
            plt.title(f'{interaction} Wqb: {round(np.min([ad[0] for ad in all_data]), 2)}')
            plt.xlabel('Distance (Angstrom)')
            plt.ylabel('Wqb (kcal/mol)')
        plt.savefig(str(Path(save_dir, f"{interaction}.png")))
        plt.close()
            
def make_wqb_df(dat_dict, sample_size=20, num_samples=10):
    np.random.seed(3)
    min_mean_dict = {}
    stdev_dict = {}
    
    for interaction, run in dat_dict.items():
        all_data=[run[i][1][0] for i in range(len(run))]
        print(all_data)
        wqb_mins = []
        try:
            for q in range(num_samples):
                sub = np.random.choice(all_data, sample_size, replace=False)
                wqb_mins.append(np.min(sub))
            mean_min = np.mean(wqb_mins)
            min_mean_dict[interaction] = mean_min
            stdev_dict[interaction] = np.array(wqb_mins).std()
        except Exception as e:
            print(e)
                
    pdb_list = []
    interactions = []
    openduck_2021_list = []
    mean_min_list = []
    stdev_list = []

    for key, run1 in dat_dict.items():
        all_dat = [run1[i][1][0] for i in range(len(run1))]
        try:
            mean_min = min_mean_dict[key]
            mean_min_list.append(mean_min)
            stdev = stdev_dict[key]
            stdev_list.append(stdev)
            openduck_2021_list.append(round(np.min([all_dat]), 2))
            pdb_id = key.split('_')[0]
            pdb_list.append(pdb_id)
            interaction = '_'.join(key.split('_')[1:])
            interactions.append(interaction)
        except KeyError as e:
            print('EXCEPTION IN SECOND LOOP')
            print(e)

    wqb_df = pd.DataFrame()
    wqb_df['PDBID'] = pdb_list
    wqb_df['Interaction'] = interactions
    wqb_df['openduck_2021'] = openduck_2021_list
    wqb_df['openduck_mean_min'] = np.round(mean_min_list, 2)
    wqb_df['mean_min_stdev'] = np.round(stdev_list, 2)

    return wqb_df

def make_all_wqb_runs_df(dat_dict):
    interact_list = []
    dat_file_list = []
    all_wqb_list = []

    for inter in dat_dict.keys():
        for di in dat_dict[inter]:
            interact_list.append(inter)
            dat_file_list.append(di[0].name)
            all_wqb_list.append(di[1][0])
    all_df = pd.DataFrame(data=np.array([interact_list, dat_file_list, all_wqb_list]).T, columns= ['Interaction', 'dat_file','Wqb'])
    return all_df

if __name__ == "__main__":
    #First, get the openduck values 
    
    all_runs = list(Path('.').glob('dat_files/*.dat'))
    to_sim = list(set([(x).name.split("_smd")[0] for x in all_runs]))
    dir_paths = [list(Path('.').glob(f'dat_files/{p}_smd_*.dat')) for p in to_sim]
    duck_dict = make_all_data_dict(dir_paths, num_runs=10)
    
    # Save the calculated values
    all_runs_df = make_all_wqb_runs_df(duck_dict)
    all_runs_df.to_csv('openduck_all_runs.csv')
    iridium_df = make_wqb_df(duck_dict, sample_size=10, num_samples=4)
    iridium_df.to_csv('openduck_values.csv')
    
    image_dir = Path('duck_images')
    make_wqb_images(duck_dict, image_dir)

    tar_dir = Path('.').resolve()
    for tp in [x for x in tar_dir.glob('*') if (x.is_dir() and '.ipy' not in x.name)]:
        try:
            with open(Path(tp, 'run.yaml')) as f:
                yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
            original_pdb = Path(tp, yaml_dict["apo_pdb_file"])
            chunk_pdb = Path(tp, "complex.pdb")
            resname = yaml_dict["prot_int"].split('_')[1]
            resnum = yaml_dict["prot_int"].split('_')[2]
            print(resname, resnum)
            chunk_resname, chunk_resnum = get_chunk_residue(original_pdb, chunk_pdb,resname, resnum)
            print(chunk_resname, chunk_resnum)

            for traj in tp.glob('duck_runs/*.dcd'):
                pdb_file = str(traj).replace('.dcd', '.pdb')
                vmd_resname = "resname " + chunk_resname + ' and  resid ' + str(chunk_resnum)
                lig_name = 'resname UNL'
                vmd_script = vmd_template(pdb_file, str(traj), lig_name, vmd_resname)
                vmd_script_path = Path(str(traj).replace('.dcd', '.vmd'))
                vmd_script_path.write_text(vmd_script)

        except Exception as e:
            print(e)
            continue
    

    
    





