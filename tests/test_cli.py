"""Tests of the command line parsing (yaml inputs, types, precedence) and of the AMBER input helpers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.openduck import parse_input  # noqa: E402
from duck.utils.amber_inputs import ligand_string_generator, residue_ranges  # noqa: E402


def write_yaml(tmp_path, text):
    path = tmp_path / 'input.yaml'
    path.write_text(text)
    return str(path)


def test_yaml_values_are_converted_with_the_option_types(tmp_path):
    yaml_file = write_yaml(tmp_path, """
ligand_mol : lig.mol
receptor_pdb : rec.pdb
interaction : A_ASN_140_ND2
small_molecule_forcefield : gaff2
water_model : SPCE
do_chunk : True
do_equilibrate : False
cutoff : 10
gpu_id : 0
HMR : yes
waters_to_retain : None
""")
    args, _ = parse_input(['openmm-prepare', '-y', yaml_file])
    assert (args.ligand, args.receptor, args.interaction) == ('lig.mol', 'rec.pdb', 'A_ASN_140_ND2')
    assert args.small_molecule_forcefield == 'GAFF2' and args.water_model == 'spce'
    assert args.do_chunk is True and args.do_equilibrate is False and args.HMR is True
    assert args.cutoff == 10.0 and args.gpu_id == 0 and args.waters_to_retain is None


def test_command_line_overrides_yaml(tmp_path):
    yaml_file = write_yaml(tmp_path, 'pickle : cs.pickle\nequilibrated_system : equil.chk\nsmd_cycles : 20\nmd_length : 0.5\n')
    args, _ = parse_input(['openmm-from-equilibrated', '-y', yaml_file, '-n', '3'])
    assert args.smd_cycles == 3 and args.md_length == 0.5 and args.pickle == 'cs.pickle'


def test_amber_float_threshold_and_list_options(tmp_path):
    yaml_file = write_yaml(tmp_path, 'ligand_mol : l.sdf\nreceptor_pdb : r.pdb\ninteraction : A_LEU_83_N\n'
                                     'wqb_threshold : 0.1\nligand_hb_elements : 7,8,16\nwater_steering : True\n')
    args, _ = parse_input(['amber-prepare', '-y', yaml_file])
    assert args.wqb_threshold == pytest.approx(0.1)          # used to be truncated to int
    assert args.ligand_hb_elements == [7, 8, 16]
    assert args.waters_to_restrain == 1                      # default with water steering


def test_missing_required_arguments(tmp_path, capsys):
    with pytest.raises(SystemExit):
        parse_input(['openmm-full-protocol', '-l', 'lig.mol'])
    assert '--interaction' in capsys.readouterr().err


def test_batch_requires_sdf():
    with pytest.raises(SystemExit):
        parse_input(['amber-prepare', '-l', 'lig.mol', '-r', 'r.pdb', '-i', 'A_LEU_83_N', '-B'])


def test_unknown_yaml_keys_are_reported(tmp_path, capsys):
    yaml_file = write_yaml(tmp_path, 'ligand_mol : l.mol\nreceptor_pdb : r.pdb\ninteraction : A_LEU_83_N\nsmd_cylces : 3\n')
    args, _ = parse_input(['openmm-full-protocol', '-y', yaml_file])
    assert 'smd_cylces' in capsys.readouterr().err and args.smd_cycles == 20


def test_engine_defaults():
    args, _ = parse_input(['openmm-full-protocol', '-l', 'l.mol', '-r', 'r.pdb', '-i', 'A_LEU_83_N'])
    assert args.precision == 'mixed' and args.integrator == 'langevin'
    assert args.force_constant_chunk == 1.0 and args.smd_spring_constant == 50.0


def test_report_defaults():
    args, _ = parse_input(['report'])
    assert args.format == 'auto' and args.step_threshold is None and args.output is sys.stdout


@pytest.mark.parametrize('numbers, expected', [([1, 2, 3, 4, 5], '1-5'), ([1, 2, 3, 5, 6], '1-3,5-6'), ([1, 3, 5], '1,3,5'),
                                               ([7], '7'), ([3, 1, 2], '1-3'), ([1, 2, 3, 10, 11], '1-3,10-11')])
def test_residue_ranges(numbers, expected):
    assert residue_ranges(numbers) == expected


def test_ligand_string_generator(tmp_path):
    sdf = tmp_path / 'ligs.sdf'
    sdf.write_text('mol1\n  body\nM  END\n$$$$\nmol2\n  body\nM  END\n')   # last record without delimiter
    blocks = list(ligand_string_generator(str(sdf)))
    assert len(blocks) == 2 and blocks[0].startswith('mol1') and blocks[1].endswith('$$$$')


def test_resume_keeps_folders_with_results(tmp_path):
    from scripts.openduck import _has_results
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'empty' / 'lig_1.mol').write_text('')
    (tmp_path / 'amber' / 'DUCK_0').mkdir(parents=True)   # finished AMBER run whose pickle was cleaned up
    (tmp_path / 'old').mkdir()
    (tmp_path / 'old' / 'cs.pickle').write_text('')       # OpenDUck <= 0.2 production folder
    assert not _has_results(str(tmp_path / 'empty'))
    assert _has_results(str(tmp_path / 'amber')) and _has_results(str(tmp_path / 'old'))


def test_waters_to_retain_none_is_accepted(tmp_path, capsys):
    from duck.steps.parametrize import load_retained_waters
    yaml_file = write_yaml(tmp_path, 'ligand_mol : l.mol\nreceptor_pdb : r.pdb\ninteraction : A_LEU_83_N\nwaters_to_retain : None\n')
    args, _ = parse_input(['openmm-prepare', '-y', yaml_file])
    assert args.waters_to_retain is None and load_retained_waters(args.waters_to_retain) is None
    assert load_retained_waters(str(tmp_path / 'missing.pdb')) is None
    pdb = tmp_path / 'w.pdb'
    pdb.write_text('HETATM    1  O   WAT A 302      -1.533  -7.345  -2.185  1.00  0.00           O\n'
                   'HETATM    2  H1  WAT A 302      -0.742  -6.811  -2.259  1.00  0.00           H\n'
                   'HETATM    3  H2  WAT A 302      -2.247  -6.709  -2.128  1.00  0.00           H\nEND\n')
    waters = load_retained_waters(str(pdb))
    assert [r.name for r in waters.residues] == ['HOH'] and 'renaming' in capsys.readouterr().err
