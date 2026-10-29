"""Generation of AMBER inputs and queue scripts for the AMBER DUck protocol.

Templates live in ``duck/templates/amber_inputs`` (pmemd inputs and NMR restraint files) and
``duck/templates/queueing_templates`` (job scripts). The protocol is minimisation, heating to
300 K in four 100 ps steps, 1 ns NPT equilibration and then cycles of 0.5 ns MD followed by SMD
at 300 K and 325 K (``jar=1``), stopping when the WQB falls below the threshold.
"""
import glob
import os
import shutil

from duck.utils.resources import data_path

# Residues excluded from the chunk restraint mask (solvent, ligand and bulk ions)
NON_CHUNK_RESIDUES = ('WAT', 'HOH', 'UNL', 'NA', 'CL')


def handle_error(error):
    """Error callback for multiprocessing pools: print the error instead of failing silently."""
    print(error, flush=True)


def write_string_to_file(file, string):
    with open(file, 'w') as fh:
        fh.write(string)


def ligand_string_generator(file):
    """Yield the molecule blocks (including the ``$$$$`` delimiter) of a multi-molecule SD file."""
    with open(file) as fh:
        mol = []
        for line in fh:
            line = line.rstrip()
            mol.append(line)
            if line == '$$$$':
                yield '\n'.join(mol)
                mol = []
        if any(x.strip() for x in mol):  # last molecule without a trailing delimiter
            yield '\n'.join(mol + ['$$$$'])


def residue_ranges(numbers):
    """Compact a list of residue numbers into an AMBER mask range string, e.g. [1, 2, 3, 5] -> '1-3,5'."""
    ranges = []
    for number in sorted(set(int(x) for x in numbers)):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in ranges)


class Queue_templates(object):
    """
    Create the job script that runs the AMBER DUck protocol.

    Attributes:
        wqb_threshold (float): WQB below which the run is stopped.
        replicas (int): number of MD/SMD cycles.
        hmr (bool): use the HMR topology.
        array_limit (int or bool): number of ligands for array jobs (False for a single job).
        top (str): topology file name used in the simulations.
        queue_dir (str): directory of the queue templates.
        commands_string (str): protocol commands inserted in the template.
        functions_string (str): bash functions inserted in the template.
    """

    def __init__(self, wqb_threshold=0, replicas=20, hmr=True, array_limit=False, keep_intermediate_files=False):
        '''
        Args:
            wqb_threshold (float): WQB below which the run is stopped.
            replicas (int): number of MD/SMD cycles.
            hmr (bool): use ``HMR_system_complex.prmtop``.
            array_limit (int or bool): number of array tasks, or False for a single job.
            keep_intermediate_files (bool): if False, the job deletes everything except the inputs,
                restraint files and SMD reports (``*.dat``) at the end.
        '''
        self.wqb_threshold = wqb_threshold
        self.replicas = replicas
        self.hmr = hmr
        self.array_limit = array_limit
        self.top = 'HMR_system_complex.prmtop' if hmr else 'system_complex.prmtop'
        self.queue_dir = data_path('templates', 'queueing_templates')
        self.commands_string = self._get_commands_string()
        if not keep_intermediate_files:
            self.commands_string += '\nfind -type f ! -regex ".*\\(dat\\|prmtop\\|inpcrd\\|in\\|dist.*rst\\|yaml\\)$" -delete\n'
        self.functions_string = self._get_functions_string()

    def _read_template(self, file):
        with open(os.path.join(self.queue_dir, file)) as f:
            return f.read()

    def _get_commands_string(self):
        cmd_template = self._read_template('commands.txt')
        return cmd_template.format(replicas=self.replicas, wqb_threshold=self.wqb_threshold, top=self.top, i='{i}')

    def _get_functions_string(self):
        funct_template = self._read_template('functions.txt')
        return funct_template.format(top=self.top, nu='{nu}', wqb_limit='{wqb_limit}')

    def write_string_to_file(self, file, string):
        write_string_to_file(file, string)

    def copy_getWqbValues_script(self):
        '''Copy the standalone WQB script used by the job to stop early.'''
        shutil.copyfile(os.path.join(self.queue_dir, 'getWqbValues.py'), 'getWqbValues.py')

    def write_queue_file(self, kind):
        '''
        Write ``duck_queue.q`` (or ``duck_array_queue.q`` for array jobs) from the ``kind`` template.

        Templates are ``{kind}.q`` and ``{kind}_array.q`` in the queueing templates directory
        (local, Slurm, SGE are provided).
        '''
        template = f'{kind}_array.q' if self.array_limit else f'{kind}.q'
        if not glob.glob(os.path.join(self.queue_dir, template)):
            raise ValueError(f'{kind} is not a stored queue template'
                             + (' or does not have an array template.' if self.array_limit else '.'))
        que_templ = self._read_template(template)
        if self.array_limit:
            self.write_string_to_file('duck_array_queue.q', que_templ.format(
                array_limit=self.array_limit, functions=self.functions_string, commands=self.commands_string))
        else:
            self.write_string_to_file('duck_queue.q', que_templ.format(
                functions=self.functions_string, commands=self.commands_string))
        self.copy_getWqbValues_script()


class Amber_templates(object):
    """
    Create the pmemd input files of the AMBER DUck protocol.

    Attributes:
        structure (parmed.Structure): prepared system (residue indices define the AMBER masks).
        interaction (tuple): (protein atom index, ligand atom index, distance), 0-based indices.
        hmr (bool): use a 4 fs time step (same simulated times).
        seed (str): random seed (``ig``), -1 for a random one.
        chunk_residues (str): mask range of the restrained (chunk) residues.
        water_mask (str): extra mask for restrained structural waters ('' if none).
    """

    def __init__(self, structure, interaction, hmr, seed='-1', waters_masked=None):
        '''
        Args:
            structure (parmed.Structure): prepared system.
            interaction (tuple): (protein atom index, ligand atom index, distance), 0-based indices.
            hmr (bool): use a 4 fs time step.
            seed (str): random seed for pmemd, '-1' for a random one.
            waters_masked (int, optional): number of (retained) waters, counted from the first one, to restrain.
        '''
        self.structure = structure
        self.interaction = interaction
        self.hmr = hmr
        self.seed = seed
        self.templates_dir = data_path('templates', 'amber_inputs')
        self.chunk_residues = self.extract_residuenumbers(structure)
        self.water_mask = self.get_restrained_waters_mask(structure, waters_masked)

    def write_string_to_file(self, file, string):
        write_string_to_file(file, string)

    def _read_template(self, file):
        with open(os.path.join(self.templates_dir, file)) as f:
            return f.read()

    def extract_residuenumbers(self, structure):
        '''AMBER mask range (1-based residue indices of the topology) of all non-solvent, non-ligand residues.'''
        return residue_ranges(r.idx + 1 for r in structure.residues if r.name not in NON_CHUNK_RESIDUES)

    def get_restrained_waters_mask(self, structure, nwaters=0):
        '''Mask addition (``| :ranges``) restraining the first ``nwaters`` waters (the retained ones), or ''.'''
        if not nwaters:
            return ''
        waters = [r.idx + 1 for r in structure.residues if r.name in ('WAT', 'HOH')][:nwaters]
        return f'| :{residue_ranges(waters)}'

    def _re_range(self, lst):
        '''Convert a list of numbers into a string of ranges (kept for backward compatibility).'''
        return residue_ranges(lst)

    def write_md_inputs(self):
        '''
        Write the inputs for minimisation (1_min.in), heating (2_heating150/200/250/300.in, 100 ps each),
        1 ns NPT equilibration (3_eq.in), 0.5 ns production MD per cycle (md.in) and the MD H-bond restraint
        (dist_md.rst).
        '''
        masks = dict(chunk_residues=self.chunk_residues, water_mask=self.water_mask)
        self.write_string_to_file('1_min.in', self._read_template('min.in').format(**masks))
        self.write_string_to_file('2_heating150.in', self._read_template('first_heating.in').format(
            seed=self.seed, **masks))
        heating_template = self._read_template('heating.in')
        for temp in ('200.0', '250.0', '300.0'):
            self.write_string_to_file(f'2_heating{temp[:3]}.in', heating_template.format(
                seed=self.seed, temp=temp, **masks))

        # same simulated times with and without HMR: 1 ns NPT equilibration, 0.5 ns MD per cycle
        time_step, eq_iterations, md_iterations = ('0.004', '250000', '125000') if self.hmr else ('0.002', '500000', '250000')
        self.write_string_to_file('3_eq.in', self._read_template('equilibration.in').format(
            seed=self.seed, time_step=time_step, iterations=eq_iterations, **masks))
        self.write_string_to_file('md.in', self._read_template('md.in').format(
            seed=self.seed, time_step=time_step, iterations=md_iterations, **masks))

        prot_idx, lig_idx = self.interaction[0], self.interaction[1]
        self.write_string_to_file('dist_md.rst', self._read_template('dist_md.rst').format(
            prot_idx=prot_idx + 1, lig_idx=lig_idx + 1))

    def write_smd_inputs(self):
        '''
        Write the SMD inputs at 300 K (duck.in) and 325 K (duck_325K.in) and the steering restraint (dist_duck.rst).
        '''
        # same simulated time (500 ps) and number of reported points (5000) with and without HMR
        time_step, iterations, savefreq = ('0.004', '125000', '25') if self.hmr else ('0.002', '250000', '50')
        smd_template = self._read_template('smd.in')
        for file, temp in (('duck.in', '300.0'), ('duck_325K.in', '325.0')):
            self.write_string_to_file(file, smd_template.format(
                temp=temp, chunk_residues=self.chunk_residues, water_mask=self.water_mask, seed=self.seed,
                time_step=time_step, iterations=iterations, savefreq=savefreq))

        prot_idx, lig_idx = self.interaction[0], self.interaction[1]
        self.write_string_to_file('dist_duck.rst', self._read_template('dist_duck.rst').format(
            prot_idx=prot_idx + 1, lig_idx=lig_idx + 1))

    def write_all_inputs(self):
        '''
        Write the md inputs and smd inputs
        '''
        self.write_md_inputs()
        self.write_smd_inputs()


if __name__ == '__main__':
    queues = Queue_templates(wqb_threshold='7', replicas='8', hmr=True, array_limit=False)
    queues.write_queue_file('local')
