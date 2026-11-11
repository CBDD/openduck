"""Independent equilibrations: several equilibrations of one prepared system, each with its own production.

All the cycles of a DUck run start from one equilibrated structure, which is also the reference of the
chunk restraints (AMBER ``-ref 3_eq.rst``). Cycles from one equilibration are therefore not independent
samples of the system: two runs that differ only in the equilibration (random seed) can disagree by more
than the error of either run. With ``--equilibrations N`` the prepared system is equilibrated N times and
each equilibration runs its own cycles in ``EQ_<k>/`` (k = 0 .. N-1):

* OpenMM: ``EQ_<k>/equil.chk`` and the production (``duck_runs/`` or ``crooks_runs/``) of that equilibration.
* AMBER: the job script runs minimisation, heating, equilibration and production inside each ``EQ_<k>/``,
  with the inputs and topology linked from the run folder.

The analysis pools all the cycles and bootstraps them in two stages (:func:`cluster_bootstrap_indices`):
first the equilibrations, then the cycles, so the error bars include the variation between equilibrations.
"""
import glob
import os
import re

import numpy as np

PREFIX = 'EQ'
_FOLDER = re.compile(rf'^{PREFIX}_(\d+)$')


def folder_name(index):
    """Folder of the equilibration ``index`` (0-based): ``EQ_<index>``."""
    return f'{PREFIX}_{index}'


def find_equilibrations(folder='.'):
    """``EQ_<k>`` folders inside ``folder`` (paths, sorted by k); empty for a run with a single equilibration."""
    found = [(int(_FOLDER.match(os.path.basename(path)).group(1)), path)
             for path in glob.glob(os.path.join(folder, f'{PREFIX}_*'))
             if os.path.isdir(path) and _FOLDER.match(os.path.basename(path))]
    return [path for _, path in sorted(found)]


def equilibration_of(path):
    """The ``EQ_<k>`` component of ``path``, or None if it does not belong to an independent equilibration."""
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if _FOLDER.match(part):
            return part
    return None


def run_independent_equilibrations(n_equilibrations, equilibrate, production, resume=False):
    """Run ``n_equilibrations`` x (equilibration + production), each in its own ``EQ_<k>/`` folder.

    Args:
        n_equilibrations (int): number of independent equilibrations.
        equilibrate (callable): called without arguments inside ``EQ_<k>/``; must write ``equil.chk``.
            Skipped if ``resume`` is set and ``equil.chk`` already exists.
        production (callable): called with the checkpoint name inside ``EQ_<k>/``. A true return value (e.g. a
            WQB below the threshold) stops the remaining equilibrations.
        resume (bool): reuse the equilibrations found in the folders.

    Returns:
        list of str: the folders that were run.
    """
    base = os.getcwd()
    folders = []
    for k in range(n_equilibrations):
        folder = folder_name(k)
        os.makedirs(folder, exist_ok=True)
        os.chdir(folder)
        try:
            print(f'Independent equilibration {k + 1} of {n_equilibrations} ({folder}/)')
            if resume and os.path.isfile('equil.chk'):
                print(f'{folder}/equil.chk already exists, skipping its equilibration (resume mode).')
            else:
                equilibrate()
            folders.append(folder)
            if production('equil.chk'):
                print(f'Stopping after {folder}: the remaining equilibrations are not run.')
                break
        finally:
            os.chdir(base)
    return folders


def cluster_bootstrap_indices(groups, size, rng, reference=None):
    """Indices of a two-stage bootstrap sample: clusters with replacement, then elements of those clusters.

    Args:
        groups (sequence): cluster label of each element (e.g. its equilibration).
        size (int): elements in the sample.
        rng (numpy.random.Generator): random generator.
        reference (sequence, optional): the resampled clusters (stage 1) to use instead of drawing new ones, e.g.
            to resample forward and reverse pulls with the same equilibrations.

    Returns:
        tuple: (indices of the sampled elements, resampled cluster labels). Elements are drawn uniformly from
        the pooled members of the resampled clusters (with their multiplicity).
    """
    groups = np.asarray(groups)
    labels = np.unique(groups)
    clusters = rng.choice(labels, size=len(labels), replace=True) if reference is None else np.asarray(reference)
    pool = np.concatenate([np.flatnonzero(groups == label) for label in clusters])
    if not len(pool):
        return np.array([], dtype=int), clusters
    return rng.choice(pool, size=size, replace=True), clusters
