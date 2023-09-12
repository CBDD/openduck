import os
import shutil

# misc
def log_result(result):
    result_list = list() # ?¿?
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

# handle raised errors
def handle_error(error):
	print(error, flush=True)

def write_string_to_file(file,string):
    with open(file, 'w') as fh:
        fh.write(string)

def ligand_string_generator(file):
    with open(file) as fh:
        mol = []
        for line in fh:
            line=line.rstrip()
            mol.append(line)
            if line == '$$$$':
                new_mol = mol
                mol = []
                yield '\n'.join(new_mol)
                
class Queue_templates(object):
    """
    A class to create queue files for Amber simulations.

Attributes:

    wqb_threshold (float): A threshold value for WQB (quasi bond work) used in simulations.
    replicas (int): The number of replicas to be used in simulations.
    hmr (bool): A boolean indicating if simulations are using HMR (Hydrogen Mass Repartitioning).
    array_limit (bool): A boolean indicating if the queue file should be generated with an array template.
    top (str): A string representing the topology file name used in simulations.
    queue_dir (str): A string representing the directory path of the queue templates.
    commands_string (str): A string representing the command line arguments used in simulations.
    functions_string (str): A string representing the Python functions used in simulations.

Methods:

    write_string_to_file(file, string): Writes a given string to a file with the given file name.
    copy_getWqbValues_script(): Copies the getWqbValues.py script from the queue templates directory to the current working directory.
    write_queue_file(kind): Generates a queue file with the given kind (template) and writes it to disk.
    """
    def __init__(self, wqb_threshold=0, replicas=20, hmr=True, array_limit=False, keep_intermediate_files=False, local_minimization=False):
        '''
        Initialize the Queue_templates class.

        Args:
            wqb_threshold (float): A threshold value for WQB (quasi bond work) used in simulations.
            replicas (int): The number of replicas to be used in simulations.
            hmr (bool): A boolean indicating if simulations are using HMR (Hydrogen Mass Repartitioning).
            array_limit (bool): A boolean indicating if the queue file should be generated with an array template.
        '''        
        self.wqb_threshold = wqb_threshold
        self.replicas = replicas
        self.hmr = hmr
        self.array_limit = array_limit
        self.local_minimization = local_minimization

        if hmr: self.top = 'HMR_system_complex.prmtop'
        else: self.top = 'system_complex.prmtop'
        
        self.queue_dir = self._get_queue_templates_dir()
        self.commands_string = self._get_commands_string()
        if not keep_intermediate_files:
            self.commands_string += '\nfind -type f ! -regex ".*\(dat\|prmtop\|inpcrd\|in\|dist.*rst\|yaml)$" -delete\n'
        self.functions_string = self._get_functions_string()
    
    def _get_queue_templates_dir(self):
        '''
        Private method to obtain queue template directories
        '''
        import duck
        p = os.path.abspath(duck.__path__[0])
        return os.path.join(p, 'templates', 'queueing_templates')
    
    def _read_template(self, file):
        '''
        Private method to read template files
        '''
        file_path = os.path.join(self.queue_dir, file)
        with open(file_path) as f:
            file_string = f.read()
        return file_string
    
    def _get_commands_string(self):
        '''
        Private method to read command template files
        '''
        cmd_template = self._read_template('commands.txt')
        return cmd_template.format(replicas=self.replicas, wqb_threshold = self.wqb_threshold, comment='#' if self.local_minimization else '', top = self.top, i='{i}')
    
    def _get_functions_string(self):
        '''
        Private method to read function template files
        '''
        funct_template = self._read_template('functions.txt')
        return funct_template.format(top=self.top, nu='{nu}', wqb_limit='{wqb_limit}')
    
    def write_string_to_file(self, file,string):
        '''
        Utils method to write strings to a file.
        '''
        with open(file, 'w') as fh:
            fh.write(string)
    def copy_getWqbValues_script(self):
        '''
        Utils method to copy the Wqb Analysis script to desired location.
        '''
        shutil.copyfile(os.path.join(self.queue_dir, 'getWqbValues.py'), 'getWqbValues.py')
    
    def write_queue_file(self, kind):
        '''
        Generates a queue file with the given kind (template) and writes it to disk.
        '''
        import glob
        if not self.array_limit and len(glob.glob(self.queue_dir+f'/{kind}.q')) == 0:
            raise ValueError(f'{kind} is not a stored queue template.')
        elif self.array_limit and len(glob.glob(self.queue_dir+f'/{kind}_array.q')) == 0:
            raise ValueError(f'{kind} is not a stored queue template or does not have an array template.')
        if self.array_limit:
            que_templ = self._read_template(os.path.join(self.queue_dir, f'{kind}_array.q'))
            self.write_string_to_file('duck_array_queue.q', que_templ.format(array_limit = self.array_limit, functions=self.functions_string, commands=self.commands_string))
        else:
            que_templ = self._read_template(os.path.join(self.queue_dir, f'{kind}.q'))
            self.write_string_to_file('duck_queue.q', que_templ.format(functions=self.functions_string, commands=self.commands_string))
        self.copy_getWqbValues_script()
        
class Amber_templates(object):
    """
    A class to create input files for Amber simulations.

    Attributes
    ----------
    structure (str): A path to the input structure file.
    interaction (tuple): A tuple containing the indices of the interacting protein and ligand atoms, and the average distance between them.
    hmr (bool): If True, use the Hydrogen Mass Repartitioning (HMR) technique.
    seed (str), optional: A seed to use for the simulations. Default is '-1'.

    Methods
    -------
    extract_residuenumbers(structure)
        Extracts the residue numbers of all protein and ligand residues from a given structure.
    write_md_inputs()
        Writes input files for the minimization, heating, equilibration, and production phases of a
        molecular dynamics simulation.
    write_smd_inputs()
        Writes input files for the steered molecular dynamics simulation.
    """
    def __init__(self, structure, interaction, hmr, seed = '-1', waters_masked = None):
        '''
        Initialize the Amber_templates class.

        Args:
           structure (str): A path to the input structure file.
           interaction (tuple): A tuple containing the indices of the interacting protein and ligand atoms, and the average distance between them.
           hmr (bool): If True, use the Hydrogen Mass Repartitioning (HMR) technique.
           seed (str), optional: A seed to use for the simulations. Default is '-1'.
        ''' 
        self.structure = structure
        self.interaction = interaction
        self.hmr = hmr
        self.seed = seed
        
        self.templates_dir = self._get_amber_templates_dir()
        self.chunk_residues = self.extract_residuenumbers(structure)
        self.water_mask = self.get_restrained_waters_mask(structure, waters_masked)

    def _get_amber_templates_dir(self):
        '''
        A private method that returns the path to the Amber templates directory.
        '''
        import duck
        p = os.path.abspath(duck.__path__[0])
        return os.path.join(p, 'templates', 'amber_inputs')
    
    def write_string_to_file(self, file,string):
        '''
        Writes a given string to a file with the given name.
        '''
        with open(file, 'w') as fh:
            fh.write(string)
            
    def _read_template(self, file):
        '''
        A private method that reads a given template file and returns its contents as a string.
        '''
        file_path = os.path.join(self.templates_dir, file)
        with open(file_path) as f:
            file_string = f.read()
        return file_string
    
    def extract_residuenumbers(self,  structure):
        r_id = []
        for r in structure.residues:
            if r.name != 'WAT' and r.name != 'UNL' and r.name != 'NA' and r.name != 'CL':
                r_id.append(int(r.number)+1)
        chunk_residues = self._re_range(r_id)
        return chunk_residues
    
    def get_restrained_waters_mask(self, structure, nwaters = 0):
        if not nwaters:
            return ''
        waters_to_mask = []
        for r in structure.residues:
            if r.name == 'WAT' or r.name == 'HOH':
                waters_to_mask.append(int(r.number)+1)
                if len(waters_to_mask) == nwaters:
                    break
        water_residues = self._re_range(waters_to_mask)
        return f'| :{water_residues}'

    def _re_range(self, lst):
        '''
        A private method that converts a list of numbers into a string of ranges.
        '''
        # from here https://stackoverflow.com/questions/9847601/convert-list-of-numbers-to-string-ranges
        # not sure if it works super well!!!!
        n = len(lst)
        result = []
        scan = 0
        while n - scan > 2:
            step = 1
            if lst[scan + 2] - lst[scan + 1] != step:
                result.append(str(lst[scan]))
                scan += 1
                continue

            for j in range(scan+2, n-1):
                if lst[j+1] - lst[j] != step:
                    result.append('{}-{}'.format(lst[scan], lst[j]+1))
                    scan = j+1
                    break
            else:
                result.append('{}-{}'.format(lst[scan], lst[n-1]))
                return ','.join(result)
        if n - scan == 1:
            result.append(str(lst[scan]))
        elif n - scan == 2:
            result.append(','.join(map(str, lst[scan:])))
        return ','.join(result)
        
    def write_md_inputs(self):
        '''
        method to generate the input files for Amber molecular dynamics simulations.
        It generates the following input files:
           1_min.in: minimization input file.
           2_heating150.in, 2_heating200.in, 2_heating250.in, 2_heating300.in: heating input files.
           3_eq.in: equilibration input file.
           md.in: production MD input file.
           dist_md.rst: restraint file for distance between the protein and ligand.
        '''
        # min
        min_template = self._read_template('min.in')
        self.write_string_to_file('1_min.in', min_template.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask))
        
        # heating
        init_heating_template = self._read_template('first_heating.in')
        self.write_string_to_file('2_heating150.in', init_heating_template.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed =self.seed ))
        
        heating_templates = self._read_template('heating.in')
        self.write_string_to_file('2_heating200.in', heating_templates.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed =self.seed, temp='200.0'))
        self.write_string_to_file('2_heating250.in', heating_templates.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed =self.seed, temp='250.0'))
        self.write_string_to_file('2_heating300.in', heating_templates.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed =self.seed, temp='300.0'))

        # to maintain the same simulation times despite hmr
        if self.hmr:
            time_step = '0.004'
            iterations= '250000'
        else:
            time_step = '0.002'
            iterations='500000'
        
        # equilibration
        equilibration_template = self._read_template('equilibration.in')
        self.write_string_to_file('3_eq.in', equilibration_template.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed =self.seed, time_step=time_step, iterations=iterations))

        # production
        md_template = self._read_template('md.in')
        self.write_string_to_file('md.in', md_template.format(chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed=self.seed, time_step=time_step, iterations=iterations))
        
        # disang
        prot_idx, lig_idx, pairmeandistance_i = self.interaction
        md_dist_template = self._read_template('dist_md.rst')
        self.write_string_to_file('dist_md.rst', md_dist_template.format(prot_idx= prot_idx+1, lig_idx=lig_idx+1))
    
    def write_smd_inputs(self):
        '''
        method to generate the input files for Amber steered molecular dynamics simulations.
        It generates the following input files:
           duck.in: SMD input file for simulations at 300K.
           duck_325K.in: SMD input file for simulations at 325K.'''
        # to maintain the same simulation times despite hmr and the smd report number
        if self.hmr:
            time_step = '0.004'
            iterations = '125000'
            savefreq = '25'
        else:
            time_step = '0.002'
            iterations = '250000'
            savefreq = '50'
        
        # smd
        smd_template = self._read_template('smd.in')
        self.write_string_to_file('duck.in', smd_template.format(temp='300.0', chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed=self.seed, time_step=time_step, iterations=iterations, savefreq=savefreq))
        self.write_string_to_file('duck_325K.in', smd_template.format(temp='325.0', chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed=self.seed, time_step=time_step, iterations=iterations, savefreq=savefreq))

        # disang
        prot_idx, lig_idx, pairmeandistance_i = self.interaction
        duck_dist_template = self._read_template('dist_duck.rst')
        self.write_string_to_file('dist_duck.rst', duck_dist_template.format(prot_idx= prot_idx+1, lig_idx=lig_idx+1))
        
    def write_all_inputs(self):
        '''
        Write the md inputs and smd inputs
        '''
        self.write_md_inputs()
        self.write_smd_inputs()
        
def copy_getWqbValues_script():
    import duck
    base_dir = os.path.abspath(duck.__path__[0])
    shutil.copyfile(os.path.join(base_dir, 'scripts', 'getWqbValues.py'), 'getWqbValues.py')
  
if __name__=='__main__':
    queues = Queue_templates(wqb_threshold='7', replicas='8', hmr=True, array_limit=False)
    queues.write_queue_file('local')
