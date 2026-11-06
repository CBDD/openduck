"""Free energies from bidirectional steered MD with the Crooks fluctuation theorem (CFT).

For a forward process A -> B (breaking the interaction, ``start_dist`` -> ``end_dist``) and its
reverse B -> A run at the same speed, the work distributions satisfy (Crooks 1999)

    P_F(W) / P_R(-W) = exp(beta (W - dG))

so ``dG`` is the work where ``P_F(W)`` and ``P_R(-W)`` cross. Estimators (kcal/mol), all computed
from the final work of each pull:

``kde``
    Gaussian kernel density estimates of ``W_F`` and ``-W_R``; ``dG`` is their crossing, the one with
    the highest density when there are several (thesis ch. 6.4). The density at the crossing measures
    the overlap of the two distributions, i.e. how much the estimate can be trusted.
``cgi``
    Crooks Gaussian intersection (Goette & Grubmüller, J. Comput. Chem. 2009): the same crossing for
    Gaussian fits of both distributions.
``bar``
    Bennett acceptance ratio, the maximum-likelihood estimator of ``dG`` from both directions
    (Shirts et al., Phys. Rev. Lett. 2003).
``jarzynski_forward`` / ``jarzynski_reverse``
    One-direction Jarzynski estimates, ``-kT ln <exp(-beta W_F)>`` and ``kT ln <exp(-beta W_R)>``.

Errors come from bootstrapping both directions independently (in two stages, equilibrations and then pulls,
for runs with independent equilibrations). The forward pulls are also DUck
SMDs, so :func:`analyse_folder` reports their WQB and DUck ``dG_QB`` (:mod:`duck.utils.analysis_and_report`)
next to the Crooks estimates.

Layouts read by :func:`find_pulls`: OpenMM ``crooks_runs/forward_<i>.dat`` and ``reverse_<i>.dat``
(:mod:`duck.steps.bidirectional`) and AMBER ``CYCLE_<i>/forward.dat`` and ``reverse.dat``
(:class:`duck.utils.amber_inputs.Crooks_amber_templates`), directly in the run folder or in the ``EQ_<k>/``
folders of independent equilibrations.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from scipy.stats import gaussian_kde

from duck.utils.analysis_and_report import (KB_KCAL, _natural_key, bootstrap_df, eprint, find_wqb_min_index,
                                            get_real_jarzynski_from_bootstrapping, normalize_by_series, read_duck_dat)
from duck.utils.equilibrations import cluster_bootstrap_indices, equilibration_of, find_equilibrations

ESTIMATORS = ('kde', 'cgi', 'bar', 'jarzynski_forward', 'jarzynski_reverse')
OPENMM_DIR = 'crooks_runs'
PROTOCOL_FILE = 'protocol.json'


# ---------------------------------------------------------------------------------------------- estimators
def jarzynski_forward(w_forward, temperature=300.0):
    """``dG = -kT ln <exp(-beta W_F)>`` (kcal/mol)."""
    kt = KB_KCAL * temperature
    w = np.asarray(w_forward, dtype=float)
    return float(-kt * (logsumexp(-w / kt) - np.log(len(w))))


def jarzynski_reverse(w_reverse, temperature=300.0):
    """``dG = kT ln <exp(-beta W_R)>`` (kcal/mol): the Jarzynski estimate of the reverse process, negated."""
    return -jarzynski_forward(w_reverse, temperature)


def bar(w_forward, w_reverse, temperature=300.0):
    """Bennett acceptance ratio estimate of ``dG`` (kcal/mol) from forward and reverse works.

    Solves ``sum_F f(M + beta (W_F - dG)) = sum_R f(-M + beta (W_R + dG))`` with the Fermi function
    ``f(x) = 1 / (1 + exp(x))`` and ``M = ln(n_F / n_R)``.
    """
    beta = 1.0 / (KB_KCAL * temperature)
    wf, wr = np.asarray(w_forward, dtype=float), np.asarray(w_reverse, dtype=float)
    m = np.log(len(wf) / len(wr))

    def imbalance(dg):  # increases monotonically with dg
        return expit(-(m + beta * (wf - dg))).sum() - expit(-(-m + beta * (wr + dg))).sum()

    low, high = min(wf.min(), -wr.max()), max(wf.max(), -wr.min())
    width = max(high - low, 1.0)
    while imbalance(low) > 0:
        low -= width
    while imbalance(high) < 0:
        high += width
    return float(brentq(imbalance, low, high, xtol=1e-8))


def gaussian_intersection(w_forward, w_reverse, temperature=None):
    """Crooks Gaussian intersection: crossing of Gaussian fits of ``W_F`` and ``-W_R`` (kcal/mol).

    With different widths the Gaussians cross twice; the crossing between both means is used (or, if
    none lies between them, the one with the highest density). ``temperature`` is accepted for a
    uniform estimator signature and not used.
    """
    if len(w_forward) < 2 or len(w_reverse) < 2:
        return float('nan')
    m1, s1 = np.mean(w_forward), np.std(w_forward, ddof=1)
    m2, s2 = np.mean(-np.asarray(w_reverse, dtype=float)), np.std(w_reverse, ddof=1)
    if not (s1 > 0 and s2 > 0):
        return float('nan')
    a = 1 / s1 ** 2 - 1 / s2 ** 2
    if abs(a) < 1e-12 * max(1 / s1 ** 2, 1 / s2 ** 2):
        return float((m1 + m2) / 2)
    b = -2 * (m1 / s1 ** 2 - m2 / s2 ** 2)
    c = m1 ** 2 / s1 ** 2 - m2 ** 2 / s2 ** 2 + 2 * np.log(s1 / s2)
    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    if not len(roots):
        return float('nan')
    between = [x for x in roots if min(m1, m2) <= x <= max(m1, m2)]
    if between:
        return float(between[0])
    return float(max(roots, key=lambda x: -((x - m1) / s1) ** 2 / 2 - np.log(s1)))


def _kde(values, bandwidth):
    '''scipy Gaussian KDE; ``bandwidth`` is a rule ("scott", "silverman") or the kernel width in kcal/mol.'''
    values = np.asarray(values, dtype=float)
    if isinstance(bandwidth, str):
        return gaussian_kde(values, bw_method=bandwidth)
    return gaussian_kde(values, bw_method=float(bandwidth) / np.std(values, ddof=1))


def kde_intersection(w_forward, w_reverse, bandwidth='scott', points=4000, min_relative_density=0.01):
    """Crossing of the kernel density estimates of ``W_F`` and ``-W_R`` (thesis ch. 6.4).

    Args:
        w_forward, w_reverse (array-like): final works of the forward and reverse pulls (kcal/mol).
        bandwidth (str or float): KDE bandwidth rule (``scott``, ``silverman``) or width in kcal/mol.
        points (int): grid points on which the densities are compared.
        min_relative_density (float): crossings whose density is below this fraction of the lower of the
            two density maxima (e.g. in the far tails) are not counted as crossings.

    Returns:
        dict: ``dG`` (work of the crossing with the highest density, kcal/mol), ``density`` (its density,
        1/(kcal/mol)), ``crossings`` (number of counted crossings), ``all_crossings`` (list of
        (work, density)) and the evaluated ``grid``, ``p_forward`` and ``p_reverse``.
    """
    forward, reverse = np.asarray(w_forward, dtype=float), -np.asarray(w_reverse, dtype=float)
    nan = dict(dG=float('nan'), density=float('nan'), crossings=0, all_crossings=[], grid=None,
               p_forward=None, p_reverse=None)
    if len(forward) < 2 or len(reverse) < 2 or np.ptp(forward) == 0 or np.ptp(reverse) == 0:
        return nan
    kde_f, kde_r = _kde(forward, bandwidth), _kde(reverse, bandwidth)
    width = np.sqrt(max(kde_f.covariance[0, 0], kde_r.covariance[0, 0]))
    grid = np.linspace(min(forward.min(), reverse.min()) - 4 * width,
                       max(forward.max(), reverse.max()) + 4 * width, points)
    p_forward, p_reverse = kde_f(grid), kde_r(grid)
    diff = p_forward - p_reverse
    crossings = []
    for i in np.nonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]:
        x = grid[i] - diff[i] * (grid[i + 1] - grid[i]) / (diff[i + 1] - diff[i])
        crossings.append((float(x), float(np.interp(x, grid, p_forward))))
    if not crossings:
        return dict(nan, grid=grid, p_forward=p_forward, p_reverse=p_reverse)
    best = max(crossings, key=lambda c: c[1])
    floor = min_relative_density * min(p_forward.max(), p_reverse.max())
    return dict(dG=best[0], density=best[1], crossings=sum(d >= floor for _, d in crossings), all_crossings=crossings,
                grid=grid, p_forward=p_forward, p_reverse=p_reverse)


def _estimate(name, wf, wr, temperature, bandwidth):
    if name == 'kde':
        return kde_intersection(wf, wr, bandwidth=bandwidth)['dG']
    if name == 'cgi':
        return gaussian_intersection(wf, wr)
    if name == 'bar':
        return bar(wf, wr, temperature)
    if name == 'jarzynski_forward':
        return jarzynski_forward(wf, temperature)
    if name == 'jarzynski_reverse':
        return jarzynski_reverse(wr, temperature)
    raise ValueError(f'Unknown estimator {name}; options: {", ".join(ESTIMATORS)}')


def crooks_estimates(w_forward, w_reverse, temperature=300.0, estimators=ESTIMATORS, bootstrap_samples=100,
                     sample_size=None, bandwidth='scott', seed=None, groups_forward=None, groups_reverse=None):
    """Free energy estimates from forward and reverse works, with bootstrap errors.

    Args:
        w_forward, w_reverse (array-like): final works of the forward and reverse pulls (kcal/mol).
        temperature (float): K.
        estimators (sequence): names from ``ESTIMATORS``.
        bootstrap_samples (int): bootstrap iterations (0 disables the errors).
        sample_size (int, optional): works drawn (with replacement) per direction and iteration;
            defaults to the number of pulls of each direction.
        bandwidth (str or float): KDE bandwidth (see :func:`kde_intersection`).
        seed (int, optional): seed of the bootstrap.
        groups_forward, groups_reverse (sequence, optional): equilibration of each pull. With more than one
            equilibration, each bootstrap iteration resamples the equilibrations first (the same ones for both
            directions) and then the pulls of those equilibrations.

    Returns:
        dict: ``{name: {'dG': estimate from all works, 'sd': bootstrap standard deviation,
        'samples': bootstrap estimates}}``. Bootstrap samples where an estimator is undefined are
        dropped.
    """
    wf, wr = np.asarray(w_forward, dtype=float), np.asarray(w_reverse, dtype=float)
    rng = np.random.default_rng(seed)
    clustered = groups_forward is not None and groups_reverse is not None and len(set(groups_forward)) > 1
    results = {}
    for name in estimators:
        samples = []
        for _ in range(bootstrap_samples):
            if clustered:
                index_f, chosen = cluster_bootstrap_indices(groups_forward, sample_size or len(wf), rng)
                index_r, _ = cluster_bootstrap_indices(groups_reverse, sample_size or len(wr), rng, reference=chosen)
                if not len(index_f) or not len(index_r):
                    continue
                sample_f, sample_r = wf[index_f], wr[index_r]
            else:
                sample_f = rng.choice(wf, size=sample_size or len(wf), replace=True)
                sample_r = rng.choice(wr, size=sample_size or len(wr), replace=True)
            samples.append(_estimate(name, sample_f, sample_r, temperature, bandwidth))
        samples = np.array([x for x in samples if np.isfinite(x)])
        results[name] = dict(dG=_estimate(name, wf, wr, temperature, bandwidth),
                             sd=float(np.std(samples)) if len(samples) > 1 else float('nan'), samples=samples)
    return results


# ---------------------------------------------------------------------------------------------- reading pulls
def find_pulls(folder='.'):
    """Forward and reverse work reports of a bidirectional run (all its ``EQ_<k>/`` folders if it has them).

    Returns:
        tuple: (layout ``'openmm'`` or ``'amber'``, forward files, reverse files), naturally sorted by
        equilibration and cycle.

    Raises:
        FileNotFoundError: if no forward and reverse reports are found.
    """
    roots = find_equilibrations(folder) or [folder]
    layouts = (('openmm', (OPENMM_DIR, 'forward_*.dat'), (OPENMM_DIR, 'reverse_*.dat')),
               ('amber', ('CYCLE_*', 'forward.dat'), ('CYCLE_*', 'reverse.dat')))
    for layout, forward_pattern, reverse_pattern in layouts:
        forward = sorted((f for root in roots for f in glob.glob(os.path.join(root, *forward_pattern))), key=_natural_key)
        reverse = sorted((f for root in roots for f in glob.glob(os.path.join(root, *reverse_pattern))), key=_natural_key)
        if forward and reverse:
            return layout, forward, reverse
    raise FileNotFoundError(f'No forward and reverse pulls found in {folder} '
                            f'({OPENMM_DIR}/forward_*.dat or CYCLE_*/forward.dat).')


def read_protocol(folder='.'):
    """``protocol.json`` of a bidirectional run (OpenMM: in ``crooks_runs/``, AMBER: in the folder), or {}."""
    roots = [folder] + find_equilibrations(folder)
    for path in [os.path.join(root, OPENMM_DIR, PROTOCOL_FILE) for root in roots] + [os.path.join(folder, PROTOCOL_FILE)]:
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh)
    return {}


def read_profiles(files):
    """Work profiles of the given reports as a DataFrame (one column per pull, ``r0`` in A as index).

    Only complete pulls are used: reports shorter than the longest one (e.g. a pmemd run still writing its
    DUMPAVE, or one that crashed) are left out with a warning, so the final works are all taken at the end
    of the pull.
    """
    data = {f: read_duck_dat(f) for f in files}
    n = max(len(d) for d in data.values())
    incomplete = [f for f, d in data.items() if len(d) < n]
    if incomplete:
        eprint(f'Warning: {len(incomplete)} unfinished pull(s) left out ({", ".join(incomplete)}).')
    complete = {f: d for f, d in data.items() if len(d) == n}
    index = np.round(next(iter(complete.values()))[:, 0], 4)
    return pd.DataFrame({f: d[:, 3] for f, d in complete.items()}, index=index)


def jarzynski_profile(profiles, temperature=300.0):
    """Row-wise Jarzynski estimate ``-kT ln <exp(-beta W)>`` of a DataFrame of work profiles."""
    kt = KB_KCAL * temperature
    work = profiles.to_numpy(dtype=float)
    return pd.Series(-kt * (logsumexp(-work / kt, axis=1) - np.log(work.shape[1])), index=profiles.index)


def duck_descriptors(forward_profiles, temperature=300.0, sample_size=20, samples=20, seed=None, groups=None):
    """DUck WQB (lowest over the forward pulls) and Jarzynski ``dG_QB`` (bootstrapped, as ``openduck report``).

    ``groups`` gives the equilibration of each pull (two-stage bootstrap, see :func:`bootstrap_df`).
    """
    wqbs = []
    for column in forward_profiles.columns:
        work = forward_profiles[column].to_numpy()
        i_min = find_wqb_min_index(work)
        wqbs.append(0.0 if i_min is None else float(np.max(work[i_min:]) - work[i_min]))
    norm = normalize_by_series(forward_profiles)
    _, sample_dfs = bootstrap_df([norm], sample_size=sample_size, samples=samples, plot=False, temps=[temperature],
                                 seed=seed, groups=[groups])
    dg, sd, _, _ = get_real_jarzynski_from_bootstrapping(sample_dfs)
    return dict(WQB=min(wqbs), dG_QB=dg, dG_QB_sd=sd)


# ---------------------------------------------------------------------------------------------- folder analysis
def analyse_folder(folder='.', temperature=None, bootstrap_samples=100, sample_size=None, bandwidth='scott', seed=None,
                   plot=False, plot_file='crooks_report.png'):
    """Crooks analysis of the bidirectional run in ``folder``.

    Args:
        folder (str): run folder (``crooks_runs/`` for OpenMM or ``CYCLE_*/`` for AMBER inside).
        temperature (float, optional): K; read from ``protocol.json`` if not given (300 K if missing).
        bootstrap_samples, sample_size, bandwidth, seed: see :func:`crooks_estimates`.
        plot (bool): write the work profiles, work distributions and free energy profiles to
            ``folder/plot_file``.

    Returns:
        dict: one report row (see ``REPORT_COLUMNS``).
    """
    layout, forward_files, reverse_files = find_pulls(folder)
    protocol = read_protocol(folder)
    if temperature is None:
        temperature = float(protocol.get('temperature', 300.0))
    forward, reverse = read_profiles(forward_files), read_profiles(reverse_files)
    wf, wr = forward.iloc[-1].to_numpy(), reverse.iloc[-1].to_numpy()
    groups_f = [equilibration_of(f) for f in forward.columns]
    groups_r = [equilibration_of(f) for f in reverse.columns]
    estimates = crooks_estimates(wf, wr, temperature, bootstrap_samples=bootstrap_samples, sample_size=sample_size,
                                 bandwidth=bandwidth, seed=seed, groups_forward=groups_f, groups_reverse=groups_r)
    kde = kde_intersection(wf, wr, bandwidth=bandwidth)
    row = dict(System=folder, engine=layout, temperature=temperature, equilibrations=len(set(groups_f)),
               n_forward=len(wf), n_reverse=len(wr))
    for name, label in (('kde', 'dG_CFT'), ('cgi', 'dG_CGI'), ('bar', 'dG_BAR'), ('jarzynski_forward', 'dG_JE_forward'),
                        ('jarzynski_reverse', 'dG_JE_reverse')):
        row[label], row[f'{label}_sd'] = estimates[name]['dG'], estimates[name]['sd']
    row.update(KDE_density=kde['density'], KDE_crossings=kde['crossings'],
               W_forward_mean=float(np.mean(wf)), W_forward_sd=float(np.std(wf, ddof=1)) if len(wf) > 1 else float('nan'),
               W_reverse_mean=float(np.mean(wr)), W_reverse_sd=float(np.std(wr, ddof=1)) if len(wr) > 1 else float('nan'),
               hysteresis=float(np.mean(wf) + np.mean(wr)))
    row.update(duck_descriptors(forward, temperature, seed=seed, groups=groups_f))
    if plot:
        plot_crooks(forward, reverse, temperature, kde, estimates, os.path.join(folder, plot_file),
                    title=os.path.basename(os.path.abspath(folder)))
    return row


REPORT_COLUMNS = ['System', 'engine', 'temperature', 'equilibrations', 'n_forward', 'n_reverse', 'dG_CFT', 'dG_CFT_sd', 'KDE_density',
                  'KDE_crossings', 'dG_CGI', 'dG_CGI_sd', 'dG_BAR', 'dG_BAR_sd', 'dG_JE_forward', 'dG_JE_forward_sd',
                  'dG_JE_reverse', 'dG_JE_reverse_sd', 'W_forward_mean', 'W_forward_sd', 'W_reverse_mean',
                  'W_reverse_sd', 'hysteresis', 'WQB', 'dG_QB', 'dG_QB_sd']


# ---------------------------------------------------------------------------------------------- plots
FORWARD_COLOR, REVERSE_COLOR = '#2a78d6', '#eb6834'
INK, INK_SECONDARY, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def plot_crooks(forward, reverse, temperature, kde, estimates, out_file, title=None):
    """Three-panel report: work profiles, work distributions with the CFT crossing and free energy profiles.

    * Work profiles: forward ``W_F(r0)`` and, for the reverse pulls, ``W_R(r0) - W_R(start)``, i.e. the
      reverse work seen from the bound state. Both start at 0; for a reversible process they coincide.
    * Distributions: kernel densities of ``W_F`` and ``-W_R`` (ticks: individual pulls) and their crossing.
    * Free energy: Jarzynski profiles of each direction referred to the bound state, and the CFT estimate
      at the end of the pull with its bootstrap error.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'axes.edgecolor': INK_SECONDARY, 'axes.labelcolor': INK, 'xtick.color': INK_SECONDARY,
                         'ytick.color': INK_SECONDARY, 'text.color': INK, 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    r_forward, r_reverse = forward.index.to_numpy(dtype=float), reverse.index.to_numpy(dtype=float)
    seen_from_start = reverse - reverse.iloc[-1]

    ax = axes[0]
    ax.plot(r_forward, forward.to_numpy(), color=FORWARD_COLOR, linewidth=0.7, alpha=0.45)
    ax.plot(r_reverse, seen_from_start.to_numpy(), color=REVERSE_COLOR, linewidth=0.7, alpha=0.45)
    ax.plot([], [], color=FORWARD_COLOR, linewidth=2, label=f'Forward (n = {forward.shape[1]})')
    ax.plot([], [], color=REVERSE_COLOR, linewidth=2, label=f'Reverse (n = {reverse.shape[1]})')
    ax.set_xlabel('Distance r0 (Å)')
    ax.set_ylabel('Work (kcal/mol)')
    ax.set_title('Work profiles', loc='left')
    ax.legend(frameon=False, loc='upper left')

    ax = axes[1]
    wf, wr = forward.iloc[-1].to_numpy(), reverse.iloc[-1].to_numpy()
    if kde['grid'] is not None:
        ax.plot(kde['grid'], kde['p_forward'], color=FORWARD_COLOR, linewidth=2, label='P$_F$(W)')
        ax.plot(kde['grid'], kde['p_reverse'], color=REVERSE_COLOR, linewidth=2, label='P$_R$(−W)')
        top = max(kde['p_forward'].max(), kde['p_reverse'].max())
    else:
        top = 1.0
    ax.plot(wf, np.full_like(wf, -0.04 * top), '|', color=FORWARD_COLOR, markersize=10)
    ax.plot(-wr, np.full_like(wr, -0.09 * top), '|', color=REVERSE_COLOR, markersize=10)
    if np.isfinite(kde['dG']):
        ax.axvline(kde['dG'], color=INK, linewidth=1, linestyle='--')
        ax.text(0.02, 0.97, f"ΔG = {kde['dG']:.1f} ± {estimates['kde']['sd']:.1f} kcal/mol\ndensity {kde['density']:.2f}",
                transform=ax.transAxes, va='top', color=INK)
    ax.set_xlabel('W (kcal/mol)')
    ax.set_ylabel('Density')
    ax.set_title('Work distributions (Crooks)', loc='left')
    ax.legend(frameon=False, loc='upper right')

    ax = axes[2]
    jarz_forward = jarzynski_profile(forward, temperature)
    jarz_reverse = jarzynski_profile(reverse, temperature)
    jarz_reverse = jarz_reverse - jarz_reverse.iloc[-1]  # G(r0) - G(start) from the reverse pulls
    ax.plot(r_forward, jarz_forward.to_numpy(), color=FORWARD_COLOR, linewidth=2, label='Jarzynski, forward')
    ax.plot(r_reverse, jarz_reverse.to_numpy(), color=REVERSE_COLOR, linewidth=2, label='Jarzynski, reverse')
    end, offset = r_forward[-1], 0.02 * abs(r_forward[-1] - r_forward[0])
    for name, marker, label, shift in (('kde', 'o', 'Crooks (KDE)', -offset), ('bar', 's', 'BAR', offset)):
        if np.isfinite(estimates[name]['dG']):
            ax.errorbar(end + shift, estimates[name]['dG'], yerr=estimates[name]['sd'], fmt=marker, color=INK, markersize=6,
                        capsize=3, markerfacecolor=INK if marker == 'o' else 'white', label=label)
    ax.set_xlabel('Distance r0 (Å)')
    ax.set_ylabel('ΔG (kcal/mol)')
    ax.set_title('Free energy', loc='left')
    ax.legend(frameon=False, loc='upper left')
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontweight='bold')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
