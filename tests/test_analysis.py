"""Tests for duck.utils.analysis_and_report (WQB and Jarzynski analysis of SMD work profiles)."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from duck.utils import analysis_and_report as ana

DATA = Path(__file__).parent / 'data'
LEGACY_DAT = DATA / 'legacy_openmm_smd_0_300.dat'
GETWQB_SCRIPT = Path(ana.__file__).parents[1] / 'templates' / 'queueing_templates' / 'getWqbValues.py'


def write_amber_dat(path, work, r0=None):
    '''Write a 4-column AMBER jar=1 style report with the given work profile.'''
    work = np.asarray(work, dtype=float)
    if r0 is None:
        r0 = np.linspace(2.5, 5.0, len(work) + 1)[1:]
    r = r0 + 0.3
    force = -100.0 * (r - r0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([r0, r, force, work]), fmt='%12.5f')
    return path


def duck_like_work(n_rows, rng, barrier=8.0, noise=0.03):
    '''A work profile with a shallow initial minimum followed by a barrier, as in a DUck SMD.'''
    x = np.linspace(0, 1, n_rows)
    return (barrier * (np.tanh((x - 0.45) * 6) + 1) / 2 - 2.0 * np.sin(np.pi * np.minimum(x / 0.3, 1))
            + np.cumsum(rng.normal(0, noise, n_rows)))


def load_getwqb_script():
    spec = importlib.util.spec_from_file_location('getWqbValues', GETWQB_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- reading SMD reports -------------------------------------------------------------------------------------

def test_read_amber_format(tmp_path):
    work = np.linspace(0, 1, 10)
    data = ana.read_duck_dat(write_amber_dat(tmp_path / 'duck.dat', work))
    assert data.shape == (10, 4)
    np.testing.assert_allclose(data[:, 3], work, atol=1e-5)
    np.testing.assert_allclose(data[:, 1] - data[:, 0], 0.3, atol=1e-5)


def test_read_legacy_openmm_format():
    data = ana.read_duck_dat(LEGACY_DAT)
    assert data.shape == (1250, 4)
    # first row: '0 2.502 A 0.3087707150363424 nm -29.28535751817121 kcal/(A mol) -0.05867071503633525 kcal/mol'
    np.testing.assert_allclose(data[0], [2.502, 3.087707150363424, -29.28535751817121, -0.05867071503633525])
    assert data[-1, 0] == pytest.approx(5.0)


@pytest.mark.parametrize('content, message', [
    ('', 'empty'),
    ('1 2 3\n4 5 6\n', 'unknown SMD report format'),
    ('1 2 3 4\n1 2 3\n', 'inconsistent'),
    ('1 2 3 x\n', 'non-numeric'),
])
def test_read_bad_formats(tmp_path, content, message):
    path = tmp_path / 'bad.dat'
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        ana.read_duck_dat(path)


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / 'duck.dat'
    path.write_text('2.5 2.8 -30.0 0.0\n\n2.6 2.8 -20.0 -0.1\n\n')
    assert ana.read_duck_dat(path).shape == (2, 4)


# --- WQB -----------------------------------------------------------------------------------------------------

def test_wqb_legacy_reference_value():
    # Reference values from the original (pre-refactor) implementation
    wqb, data, wqb_min = ana.get_Wqb_value(LEGACY_DAT, mode='openmm')
    assert wqb == pytest.approx(13.100706844193414, abs=1e-12)
    assert wqb_min == pytest.approx(-3.7587628367908485, abs=1e-12)
    assert ana.get_Wqb_value(LEGACY_DAT)[0] == wqb  # mode is only a hint


def test_wqb_minimum_at_first_point(tmp_path):
    # Work increasing from the start: the minimum is the first point and must not be reported as "not found"
    work = np.linspace(0, 6, 1000) ** 1.5
    wqb, data, wqb_min = ana.get_Wqb_value(write_amber_dat(tmp_path / 'duck.dat', work))
    assert wqb_min == pytest.approx(0.0, abs=1e-5)
    assert wqb == pytest.approx(work.max(), abs=1e-4)


def test_wqb_local_minimum(tmp_path):
    rng = np.random.default_rng(0)
    work = duck_like_work(5000, rng, noise=0.0)
    wqb, _, wqb_min = ana.get_Wqb_value(write_amber_dat(tmp_path / 'duck.dat', work))
    i_min = int(np.argmin(work[:2500]))
    assert wqb_min == pytest.approx(work[i_min], abs=1e-5)
    assert wqb == pytest.approx(work[i_min:].max() - work[i_min], abs=1e-4)


def test_wqb_no_minimum_found(tmp_path):
    # Profiles shorter than one 200-point segment cannot be searched
    wqb, data, wqb_min = ana.get_Wqb_value(write_amber_dat(tmp_path / 'duck.dat', np.linspace(1, 0, 150)))
    assert (wqb, wqb_min) == (0, 0)
    assert data.shape == (150, 4)


def test_find_wqb_min_index_matches_window_rule():
    work = np.zeros(1000)
    work[350] = -1.0  # a flat start is already a local minimum (first point of the first window)
    assert ana.find_wqb_min_index(work) == 0
    work = np.full(1000, 1.0)
    work[:300] = np.linspace(1.0, 0.0, 300)  # decreasing until 299, flat afterwards
    work[299] = -1.0
    assert ana.find_wqb_min_index(work) == 299


def test_standalone_getwqb_script_matches_library(tmp_path, monkeypatch):
    script = load_getwqb_script()
    rng = np.random.default_rng(1)
    for i in range(3):
        write_amber_dat(tmp_path / f'DUCK_{i}' / 'duck.dat', duck_like_work(5000, rng))
    write_amber_dat(tmp_path / 'DUCK_325K_0' / 'duck.dat', np.linspace(0, 6, 5000))  # minimum at index 0
    monkeypatch.chdir(tmp_path)
    lib_wqb, per_replica = ana.get_Wqb_value_AMBER_all()
    assert script.get_Wqb_value_AMBER_all() == pytest.approx(lib_wqb, abs=1e-12)
    assert [name for name, _ in per_replica] == ['DUCK_0', 'DUCK_1', 'DUCK_2', 'DUCK_325K_0']
    assert dict(per_replica)['DUCK_325K_0'] == pytest.approx(6.0, abs=1e-4)


def test_openmm_all_natural_order(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    for cycle in (0, 2, 10):
        write_amber_dat(tmp_path / 'duck_runs' / f'smd_{cycle}_300.dat', duck_like_work(1000, rng))
    monkeypatch.chdir(tmp_path)
    wqb, values = ana.get_Wqb_value_Openmm_all(plot=True)
    assert [Path(name).name for name, _ in values] == ['smd_0_300.dat', 'smd_2_300.dat', 'smd_10_300.dat']
    assert wqb == min(v for _, v in values)
    assert (tmp_path / 'wqb_plot.png').exists()


def test_wqb_all_without_replicas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ana.get_Wqb_value_AMBER_all() == ('Nan', [])
    assert ana.get_Wqb_value_Openmm_all() == ('Nan', [])


# --- temperature classification ------------------------------------------------------------------------------

@pytest.mark.parametrize('path, temperature', [
    ('DUCK_0/duck.dat', 300),
    ('DUCK_325/duck.dat', 300),  # replica 325 at 300 K
    ('DUCK_325K_3/duck.dat', 325),
    ('run/DUCK_325K_12/duck.dat', 325),
    ('duck_runs/smd_3_300.dat', 300),
    ('duck_runs/smd_325_300.dat', 300),  # cycle 325 at 300 K
    ('duck_runs/smd_0_325.dat', 325),
    ('somewhere/custom.dat', None),
])
def test_smd_temperature(path, temperature):
    assert ana.smd_temperature(path) == temperature


def test_read_duckdat_filters_temperature(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    for folder in ('DUCK_0', 'DUCK_325', 'DUCK_325K_0'):
        write_amber_dat(tmp_path / folder / 'duck.dat', duck_like_work(400, rng))
    monkeypatch.chdir(tmp_path)
    cold = ana.read_DUckdat(temp=300, pattern='DUCK_*/duck.dat')
    hot = ana.read_DUckdat(temp=325, pattern='DUCK_*/duck.dat')
    assert sorted(cold.columns) == ['DUCK_0/duck.dat', 'DUCK_325/duck.dat']
    assert list(hot.columns) == ['DUCK_325K_0/duck.dat']
    assert cold.shape == (400, 2) and cold.index[0] == str(round(float('2.50625'), 4))
    assert ana.read_DUckdat(temp=310, pattern='DUCK_*/duck.dat').empty


def test_read_duckdat_rejects_different_lengths(tmp_path, monkeypatch):
    write_amber_dat(tmp_path / 'DUCK_0' / 'duck.dat', np.zeros(300))
    write_amber_dat(tmp_path / 'DUCK_1' / 'duck.dat', np.zeros(200))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='same length'):
        ana.read_DUckdat(temp=300, pattern='DUCK_*/duck.dat')


# --- Jarzynski -----------------------------------------------------------------------------------------------

def load_expavg_input():
    work = pd.read_csv(DATA / 'expavg_input.csv', index_col=0)
    work.index = work.index.astype(str)
    return work


@pytest.mark.parametrize('temperature', [300, 325])
def test_expavg_fd_matches_reference(temperature):
    # Reference generated with the original row-by-row implementation (real work data + edge cases)
    reference = pd.read_csv(DATA / f'expavg_reference_{temperature}K.csv', index_col=0)
    result = ana.get_expavg_FD_df(load_expavg_input(), T=temperature)
    assert list(result.columns) == list(reference.columns)
    np.testing.assert_allclose(result.to_numpy(), reference.to_numpy(), rtol=1e-10, atol=1e-12)


def test_expavg_without_fd():
    result = ana.get_expavg_FD_df(load_expavg_input(), T=300, calculate_FD=False)
    reference = pd.read_csv(DATA / 'expavg_reference_300K.csv', index_col=0)
    np.testing.assert_allclose(result[['expavg', 'avg', 'variance']], reference[['expavg', 'avg', 'variance']], rtol=1e-10)
    assert result.drop(columns=['expavg', 'avg', 'variance']).isna().all().all()


def test_expavg_single_replica_and_large_work():
    work = pd.DataFrame({'a': [0.0, 2.0, 600.0]}, index=['2.5', '2.6', '2.7'])
    result = ana.get_expavg_FD_df(work, T=300)
    np.testing.assert_allclose(result['expavg'], work['a'])  # no underflow of exp(-B W)
    np.testing.assert_allclose(result[['Wdis', 'MSE', 'sqrtMSE_FD']], 0.0)


def test_normalize_default_is_first_half():
    work = pd.DataFrame({'a': [3.0, 1.0, 2.0, -5.0], 'b': [0.0, 4.0, -1.0, 2.0]})
    normalized = ana.normalize_by_series(work)
    np.testing.assert_allclose(normalized['a'], [2.0, 0.0, 1.0, -6.0])
    np.testing.assert_allclose(normalized['b'], [0.0, 4.0, -1.0, 2.0])
    np.testing.assert_allclose(ana.normalize_by_series(work, index_threshold=3)['b'], [1.0, 5.0, 0.0, 3.0])
    # A threshold longer than the profile falls back to the first half
    np.testing.assert_allclose(ana.normalize_by_series(work, index_threshold=2500), normalized)


def test_average_dataframes():
    one = pd.DataFrame({'x': [1.0, np.nan]}, index=['a', 'b'])
    two = pd.DataFrame({'x': [3.0, 1.0]}, index=['a', 'b'])
    np.testing.assert_allclose(ana.average_dataframes(one, two)['x'], [2.0, np.nan])
    assert ana.average_dataframes(one, pd.DataFrame()) is one


def test_bootstrap_is_reproducible_with_seed():
    work = load_expavg_input().iloc[:250]
    first = ana.bootstrap_df([work, work], sample_size=5, samples=3, plot=False, seed=42)
    second = ana.bootstrap_df([work, work], sample_size=5, samples=3, plot=False, seed=42)
    assert len(first[1]) == 6 and len(first[0]) == 6 * 250
    pd.testing.assert_frame_equal(first[0], second[0])
    stats = ana.get_stats_from_bootstrapping(first[0], list(work.index))
    assert list(stats.columns) == ['CV', 'AvgJarzynski', 'sd', 'sem'] and len(stats) == 250
    sub = first[0][first[0]['CV'] == float(work.index[10])]['Jarzynski']
    assert stats['AvgJarzynski'][10] == pytest.approx(sub.mean())
    assert stats['sd'][10] == pytest.approx(np.std(sub))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')  # SEM of a single sample is NaN
def test_real_jarzynski_max_after_min():
    profile = pd.DataFrame({'expavg': [5.0, -1.0, 0.0, 3.0, 2.0, 1.0]})
    avg, sd, sem_v, values = ana.get_real_jarzynski_from_bootstrapping([profile, profile])
    assert values == [3.0, 3.0] and avg == 3.0 and sd == 0.0  # minimum searched in the first half
    # with the search window restricted to the first point, the maximum after it is the first point itself
    assert ana.get_real_jarzynski_from_bootstrapping([profile], splitting_point=1)[3] == [5.0]


@pytest.mark.parametrize('layout', ['amber', 'openmm'])
def test_jarzynski_end_to_end(tmp_path, monkeypatch, layout):
    rng = np.random.default_rng(5)
    for i in range(5):
        for temperature in (300, 325):
            work = duck_like_work(600, rng)
            if layout == 'amber':
                path = tmp_path / (f'DUCK_{i}' if temperature == 300 else f'DUCK_325K_{i}') / 'duck.dat'
            else:
                path = tmp_path / 'duck_runs' / f'smd_{i}_{temperature}.dat'
            write_amber_dat(path, work)
    monkeypatch.chdir(tmp_path)
    avg, sd, sem_v = ana.do_jarzynski_analysis(sample_size=5, samples=4, plot=True, mode=None, seed=0)
    assert np.isfinite([avg, sd, sem_v]).all() and avg > 0
    for name in ('wqb_plot_jarzynski.png', 'bootstraped_WQB_plot.png', 'wqb_hist.pdf', 'fluctuation_dissipation.csv',
                 'bootstrapped_stats.csv', 'jarz_sd_sem.tbl'):
        assert (tmp_path / name).exists(), name
    fd = pd.read_csv(tmp_path / 'fluctuation_dissipation.csv', index_col=0)
    assert fd.shape == (600, 12)
    saved = [float(x) for x in (tmp_path / 'jarz_sd_sem.tbl').read_text().split('\t')]
    np.testing.assert_allclose(saved, [avg, sd, sem_v])


def test_jarzynski_without_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='No SMD work reports'):
        ana.do_jarzynski_analysis(plot=False)


# --- report tables -------------------------------------------------------------------------------------------

def test_build_report_df_modes():
    info = {'sysB': [2.0, [('DUCK_0', 2.0), ('DUCK_1', 4.0)]], 'sysA': [1.0, [('DUCK_0', 1.0), ('DUCK_1', 3.0)]]}
    assert ana.build_report_df(info, mode='min')['System'].tolist() == ['sysA', 'sysB']
    avg = ana.build_report_df(info, mode='avg').set_index('System')
    assert avg.loc['sysB', 'Average'] == 3.0 and avg.loc['sysB', 'SD'] == 1.0
    single = ana.build_report_df(info, mode='single')
    assert list(single.columns) == ['System', 'DUCK_0', 'DUCK_1']
    full = {k: v + [5.0, 0.5, 0.1] for k, v in info.items()}
    assert list(ana.build_report_df(full, mode='all').columns) == [
        'System', 'WQB', 'Average', 'SD', 'Jarzynski', 'Jarzynski_SD', 'Jarzynski_SEM', 'DUCK_0', 'DUCK_1']
    with pytest.raises(ValueError):
        ana.build_report_df(info, mode='nope')
