#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print the lowest WQB of the AMBER DUck replicas in the current directory.

Standalone (numpy only) copy of the WQB calculation in duck.utils.analysis_and_report, so it can run on
cluster nodes where OpenDUck is not installed. It is copied next to the queue scripts and used by their
check_WQB function to stop a DUck run early. Only stdout carries the result; warnings go to stderr.

Usage: python getWqbValues.py [folder_prefix (default DUCK) [report_name (default duck.dat)]]
"""
import os
import sys

import numpy as np


def find_wqb_min_index(work, segment=200):
    '''First local minimum of the work profile (searched by segments of 200 points), or None.'''
    half = segment // 2
    for i_segment in range(len(work) // segment):
        index_global = int(np.argmin(work[i_segment * segment:(i_segment + 1) * segment])) + i_segment * segment
        window_start = index_global - half if index_global > half else 0
        index_local = int(np.argmin(work[window_start:index_global + half + 1]))
        if index_global < half:
            return index_global if index_local == index_global else None
        if index_local == half:
            return index_global
    return None


def get_Wqb_value_AMBER(file_duck_dat):
    '''WQB of one AMBER jar=1 report (columns r0, r, force, work); 0 if no work minimum is found.'''
    data = np.loadtxt(file_duck_dat, ndmin=2)
    if data.shape[1] != 4:
        raise ValueError(f'{file_duck_dat} does not have the 4 columns of an AMBER DUck report.')
    work = data[:, 3]
    wqb_min_index = find_wqb_min_index(work)
    if wqb_min_index is None:
        sys.stderr.write(f'Warning: the work minimum could not be found in {file_duck_dat}, setting its WQB to 0\n')
        return 0.0, data, 0.0
    wqb_min = work[wqb_min_index]
    return np.max(work[wqb_min_index:]) - wqb_min, data, wqb_min


def get_Wqb_value_AMBER_all(prefix='DUCK', file='duck.dat'):
    '''Lowest WQB over all <prefix>* folders of the current directory that contain `file`.'''
    wqb_values = []
    for fol in sorted(os.listdir(os.getcwd())):
        dat_file = os.path.join(fol, file)
        if fol.startswith(prefix) and os.path.isfile(dat_file):
            wqb_values.append(get_Wqb_value_AMBER(dat_file)[0])
    if not wqb_values:
        raise FileNotFoundError(f'No {prefix}*/{file} reports found in {os.getcwd()}')
    return min(wqb_values)


if __name__ == '__main__':
    try:
        print(get_Wqb_value_AMBER_all(*sys.argv[1:3]))
    except (OSError, ValueError) as error:
        sys.stderr.write(f'Error: {error}\n')
        sys.exit(1)
