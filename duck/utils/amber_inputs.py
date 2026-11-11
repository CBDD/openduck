"""Generation of AMBER inputs and queue scripts for the AMBER DUck protocol.

Templates live in ``duck/templates/amber_inputs`` (pmemd inputs and NMR restraint files) and
``duck/templates/queueing_templates`` (job scripts). The protocol is minimisation, heating to
300 K in four 100 ps steps, 1 ns NPT equilibration and then cycles of 0.5 ns MD followed by SMD
at 300 K and 325 K (``jar=1``), stopping when the WQB falls below the threshold.

``Crooks_amber_templates`` and ``Crooks_queue_templates`` write the bidirectional variant (forward
and reverse pulls in ``CYCLE_<i>/``, see :mod:`duck.steps.bidirectional` for the OpenMM version).

With ``equilibrations > 1`` the job script repeats the whole protocol (from the minimisation) in ``EQ_<k>/``
folders, each an independent equilibration with its own production (:mod:`duck.utils.equilibrations`).
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
        replicas (int): number of MD/SMD cycles (per equilibration).
        hmr (bool): use the HMR topology.
        array_limit (int or bool): number of ligands for array jobs (False for a single job).
        equilibrations (int): independent equilibrations, each with ``replicas`` cycles in ``EQ_<k>/``.
        top (str): topology file name used in the simulations.
        queue_dir (str): directory of the queue templates.
        commands_string (str): protocol commands inserted in the template.
        functions_string (str): bash functions inserted in the template.
    """

    def __init__(self, wqb_threshold=0, replicas=20, hmr=True, array_limit=False, keep_intermediate_files=False,
                 equilibrations=1):
        '''
        Args:
            wqb_threshold (float): WQB below which the run is stopped.
            replicas (int): number of MD/SMD cycles (per equilibration).
            hmr (bool): use ``HMR_system_complex.prmtop``.
            array_limit (int or bool): number of array tasks, or False for a single job.
            keep_intermediate_files (bool): if False, the job deletes everything except the inputs,
                restraint files and SMD reports (``*.dat``) at the end.
            equilibrations (int): independent equilibrations. With more than one, the protocol runs in
                ``EQ_<k>/`` folders (k = 0 .. equilibrations - 1), each from the minimisation; the inputs must use
                random seeds (``ig=-1``). A WQB below the threshold stops the whole job.
        '''
        self.wqb_threshold = wqb_threshold
        self.replicas = replicas
        self.hmr = hmr
        self.array_limit = array_limit
        self.equilibrations = int(equilibrations)
        self.top = 'HMR_system_complex.prmtop' if hmr else 'system_complex.prmtop'
        self.queue_dir = data_path('templates', 'queueing_templates')
        self.commands_string = self._get_commands_string()
        if self.equilibrations > 1:
            self.commands_string = self._read_template('equilibrations.txt').format(
                equilibrations=self.equilibrations, top=self.top, commands=self.commands_string)
        if not keep_intermediate_files:
            self.commands_string += '\nfind -type f ! -regex ".*\\(dat\\|prmtop\\|inpcrd\\|in\\|dist.*rst\\|yaml\\|json\\)$" -delete\n'
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

    def __init__(self, structure, interaction, hmr, seed='-1', waters_masked=None, md_length=0.5, temperature=300.0,
                 md_restraint=(2.0, 3.0, 4.0, 1.0, 10.0)):
        '''
        Args:
            structure (parmed.Structure): prepared system.
            interaction (tuple): (protein atom index, ligand atom index, distance), 0-based indices.
            hmr (bool): use a 4 fs time step.
            seed (str): random seed for pmemd, '-1' for a random one.
            waters_masked (int, optional): number of (retained) waters, counted from the first one, to restrain.
            md_length (float): production MD per cycle in ns.
            temperature (float): temperature of the production MD in K (heating and equilibration reach 300 K).
            md_restraint (sequence): flat-bottom restraint of the interaction during MD (``dist_md.rst``):
                ``(r2, r3, r4, rk2, rk3)`` in A and kcal/mol/A^2.
        '''
        self.structure = structure
        self.interaction = interaction
        self.hmr = hmr
        self.seed = seed
        self.md_length = float(md_length)
        self.temperature = float(temperature)
        self.md_restraint = tuple(float(x) for x in md_restraint)
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

    @property
    def time_step(self):
        '''Time step in ps (4 fs with HMR).'''
        return 0.004 if self.hmr else 0.002

    def steps(self, length_ps):
        '''Number of steps of a run of ``length_ps``.'''
        return int(round(length_ps / self.time_step))

    def write_md_inputs(self):
        '''
        Write the inputs for minimisation (1_min.in), heating (2_heating150/200/250/300.in, 100 ps each),
        1 ns NPT equilibration (3_eq.in), production MD per cycle (md.in, 0.5 ns by default) and the MD
        H-bond restraint (dist_md.rst).
        '''
        masks = dict(chunk_residues=self.chunk_residues, water_mask=self.water_mask)
        self.write_string_to_file('1_min.in', self._read_template('min.in').format(**masks))
        self.write_string_to_file('2_heating150.in', self._read_template('first_heating.in').format(
            seed=self.seed, **masks))
        heating_template = self._read_template('heating.in')
        for temp in ('200.0', '250.0', '300.0'):
            self.write_string_to_file(f'2_heating{temp[:3]}.in', heating_template.format(
                seed=self.seed, temp=temp, **masks))

        # same simulated times with and without HMR: 1 ns NPT equilibration, md_length ns MD per cycle
        time_step = f'{self.time_step:g}'
        self.write_string_to_file('3_eq.in', self._read_template('equilibration.in').format(
            seed=self.seed, time_step=time_step, iterations=self.steps(1000.0), **masks))
        self.write_string_to_file('md.in', self._read_template('md.in').format(
            seed=self.seed, time_step=time_step, iterations=self.steps(self.md_length * 1000.0), temp=self.temperature,
            length_ns=self.md_length, **masks))

        prot_idx, lig_idx = self.interaction[0], self.interaction[1]
        r2, r3, r4, rk2, rk3 = self.md_restraint
        self.write_string_to_file('dist_md.rst', self._read_template('dist_md.rst').format(
            prot_idx=prot_idx + 1, lig_idx=lig_idx + 1, r2=r2, r3=r3, r4=r4, rk2=rk2, rk3=rk3))

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


class Crooks_amber_templates(Amber_templates):
    """
    pmemd inputs of the bidirectional protocol: DUck preparation and MD, then in each cycle an optional
    approach to ``start_dist``, a forward pull to ``end_dist`` and a reverse pull back to ``start_dist`` with
    new velocities (``irest=0``). Files: ``approach.in``/``dist_approach.rst`` (the job script replaces
    ``CURRENT_DISTANCE`` by the distance at the end of the MD), ``forward.in``/``dist_forward.rst`` and
    ``reverse.in``/``dist_reverse.rst``.
    """

    def __init__(self, structure, interaction, hmr, start_dist=2.5, end_dist=5.0, speed=5.0, spring_k=50.0,
                 approach_length=0.1, approach_spring_k=None, trajectory_interval=0.0, **kwargs):
        '''
        Args:
            structure, interaction, hmr: see :class:`Amber_templates` (also ``seed``, ``waters_masked``,
                ``md_length``, ``temperature`` and ``md_restraint`` as keyword arguments).
            start_dist, end_dist (float): ends of the pull in A.
            speed (float): steering speed in A/ns.
            spring_k (float): steering spring constant (rk2) in kcal/mol/A^2.
            approach_length (float): ns to move the spring from the current distance to ``start_dist`` before
                each forward pull (0: the forward pull starts from the MD, as a DUck SMD).
            approach_spring_k (float, optional): spring constant of the approach (default ``spring_k``).
            trajectory_interval (float): ps between saved frames of the pulls (0: no trajectories).
        '''
        super().__init__(structure, interaction, hmr, **kwargs)
        self.start_dist, self.end_dist = float(start_dist), float(end_dist)
        self.speed = float(speed)
        self.spring_k = float(spring_k)
        self.approach_length = float(approach_length)
        self.approach_spring_k = float(approach_spring_k or spring_k)
        self.trajectory_interval = float(trajectory_interval)

    def _pull_input(self, title, length_ps, dumpave, disang, reassign_velocities=False):
        # irest=0 with ntx=1 reads only the coordinates and draws new velocities at tempi; always with a random
        # seed, otherwise a fixed seed would give every reverse pull the same velocities
        ntx, irest, tempi = (1, 0, f' tempi={self.temperature:.1f},') if reassign_velocities else (5, 1, '')
        seed = '-1' if reassign_velocities else self.seed
        return self._read_template('pull.in').format(
            title=title, ntx=ntx, irest=irest, tempi=tempi, temp=self.temperature, iterations=self.steps(length_ps), time_step=f'{self.time_step:g}',
            ntwx=self.steps(self.trajectory_interval) if self.trajectory_interval else 0, seed=seed,
            chunk_residues=self.chunk_residues, water_mask=self.water_mask, savefreq=self.steps(0.1), disang=disang,
            dumpave=dumpave)

    def _restraint(self, title, r_start, r_end, rk2):
        prot_idx, lig_idx = self.interaction[0], self.interaction[1]
        return self._read_template('dist_pull.rst').format(title=title, prot_idx=prot_idx + 1, lig_idx=lig_idx + 1,
                                                          r_start=r_start, r_end=r_end, rk2=rk2)

    def write_smd_inputs(self):
        '''Write the approach, forward and reverse inputs and their steering restraints.'''
        pull_ps = abs(self.end_dist - self.start_dist) / self.speed * 1000.0
        a, b = f'{self.start_dist:g}', f'{self.end_dist:g}'
        if self.approach_length > 0:
            self.write_string_to_file('approach.in', self._pull_input(
                f'Approach to {a} A in {self.approach_length:g} ns', self.approach_length * 1000.0, 'approach.dat',
                'dist_approach.rst'))
            self.write_string_to_file('dist_approach.rst', self._restraint(
                f'move the spring from the current distance to {a} A', 'CURRENT_DISTANCE', self.start_dist,
                self.approach_spring_k))
        self.write_string_to_file('forward.in', self._pull_input(
            f'Forward pull from {a} to {b} A ({pull_ps / 1000:g} ns, {self.speed:g} A/ns)', pull_ps, 'forward.dat',
            '../dist_forward.rst'))
        self.write_string_to_file('reverse.in', self._pull_input(
            f'Reverse pull from {b} to {a} A ({pull_ps / 1000:g} ns, {self.speed:g} A/ns) with new velocities', pull_ps,
            'reverse.dat', '../dist_reverse.rst', reassign_velocities=True))
        self.write_string_to_file('dist_forward.rst', self._restraint(
            f'forward pull from {a} to {b} A', f'{self.start_dist:.2f}', self.end_dist, self.spring_k))
        self.write_string_to_file('dist_reverse.rst', self._restraint(
            f'reverse pull from {b} to {a} A', f'{self.end_dist:.2f}', self.start_dist, self.spring_k))


class Crooks_queue_templates(Queue_templates):
    """Job script of the bidirectional AMBER protocol (``crooks_commands.txt`` and ``crooks_functions.txt``)."""

    def __init__(self, interaction, approach=True, wqb_threshold=0, replicas=20, hmr=True, array_limit=False,
                 keep_intermediate_files=False, equilibrations=1):
        '''
        Args:
            interaction (tuple): (protein atom index, ligand atom index, ...), 0-based (for the cpptraj distance
                measured before the approach).
            approach (bool): the inputs include an approach phase.
            wqb_threshold, replicas, hmr, array_limit, keep_intermediate_files, equilibrations: see
                :class:`Queue_templates` (``replicas`` is the number of cycles per equilibration).
        '''
        self.prot_idx, self.lig_idx = int(interaction[0]) + 1, int(interaction[1]) + 1
        self.approach = approach
        super().__init__(wqb_threshold=wqb_threshold, replicas=replicas, hmr=hmr, array_limit=array_limit,
                         keep_intermediate_files=keep_intermediate_files, equilibrations=equilibrations)

    def _get_commands_string(self):
        return self._read_template('crooks_commands.txt').format(cycles=self.replicas, wqb_threshold=self.wqb_threshold,
                                                                 top=self.top)

    def _get_functions_string(self):
        return self._read_template('crooks_functions.txt').format(top=self.top, prot_idx=self.prot_idx,
                                                                  lig_idx=self.lig_idx, approach=int(self.approach))


if __name__ == '__main__':
    queues = Queue_templates(wqb_threshold='7', replicas='8', hmr=True, array_limit=False)
    queues.write_queue_file('local')
