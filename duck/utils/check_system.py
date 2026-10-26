"""Sanity checks on OpenMM StateDataReporter logs (equilibration and MD stability)."""
import csv
import sys

import numpy as np

POTENTIAL_ENERGY_COLUMN = 'Potential Energy (kJ/mole)'
TEMPERATURE_COLUMN = 'Temperature (K)'


def check_pot_eng(pot_eng):
    """Heuristic stability check of a potential energy series (kJ/mol).

    The system is flagged if the potential energy is not clearly negative or fluctuates
    abnormally. The thresholds are empirical (inherited from the original DUck pipeline).

    Args:
        pot_eng (sequence of float): potential energy values in kJ/mol.

    Returns:
        bool: True if the series looks stable.
    """
    pot_eng = np.asarray(pot_eng, dtype=float)
    if pot_eng.size == 0:
        print('NO POTENTIAL ENERGY DATA')
        return False
    if pot_eng.max() > -9876.0:
        print('VERY HIGH POTENTIAL ENERGY')
        return False
    if pot_eng.std() > 5000:
        print('HIGHLY VOLATILE')
        return False
    if pot_eng.size > 1 and np.abs(np.diff(pot_eng)).max() > 20000:
        print('HIGHLY VOLATILE')
        return False
    return True


def _read_column(input_file, column, header=True, delimiter=','):
    """Read one column of a CSV log, by name (if ``column`` is a str and header=True) or index."""
    with open(input_file) as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        return []
    if header:
        names, rows = [x.strip('#"\' ') for x in rows[0]], rows[1:]
        if isinstance(column, str):
            column = names.index(column)
    return [float(row[column]) for row in rows if row]


def check_if_equlibrated(input_file, column, header=True, delimiter=","):
    """Check the potential energy stored in an OpenMM log.

    Args:
        input_file (str): StateDataReporter CSV file.
        column (int or str): index or name of the potential energy column. When the file has
            a header containing ``Potential Energy (kJ/mole)`` that column is always used.
        header (bool): whether the first line is a header.
        delimiter (str): column delimiter.

    Returns:
        bool: True if the system looks stable.
    """
    if header:
        with open(input_file) as fh:
            first_line = fh.readline()
        if POTENTIAL_ENERGY_COLUMN in first_line:
            column = POTENTIAL_ENERGY_COLUMN
    return check_pot_eng(_read_column(input_file, column, header, delimiter))


def check_temperature(input_file, target, tolerance=10.0):
    """Warn if the mean temperature in an OpenMM log deviates from the thermostat target.

    Args:
        input_file (str): StateDataReporter CSV file with a temperature column.
        target (float): thermostat temperature in K.
        tolerance (float): allowed deviation of the mean temperature in K.

    Returns:
        bool: True if the mean temperature is within tolerance (or there is no data).
    """
    try:
        temperatures = _read_column(input_file, TEMPERATURE_COLUMN)
    except (ValueError, IndexError, OSError):
        return True
    if not temperatures:
        return True
    mean = float(np.mean(temperatures))
    if abs(mean - target) > tolerance:
        sys.stderr.write(f'WARNING: mean temperature in {input_file} is {mean:.1f} K (thermostat at {target:g} K). '
                         'The simulation may be unstable or the GPU results corrupted.\n')
        return False
    return True
