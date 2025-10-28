import parmed as pmd
import argparse
import sys
def frcmod_mol2_to_xml(frcmod, mol2, out_xml):
	# Load the Amber FF
	ff = pmd.openmm.OpenMMParameterSet.from_parameterset(pmd.amber.AmberParameterSet(frcmod))

	# Load mol2 files
	mol2 = pmd.load_file(mol2)

	# If there are multiple residue definitions, mol2 will be a pmd.modeller.ResidueTemplateContainer
	# Otherwise it will be a pmd.modeller.ResidueTemplate
	if isinstance(mol2, pmd.modeller.ResidueTemplateContainer):
		ff.residues = mol2.to_library()
	else:
		ff.residues[mol2.name] = mol2
	ff.write(out_xml)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FRCMOD + mol2 to xlm ff')
	parser.add_argument('--frcmod', '-f', required=True, type=str, help='Frcmod with all information, not just what is missing in gaff or parm10')
	parser.add_argument('--mol2', '-m', required=True, type=str, help='Mol2 of the molecule with the partial charges.')
	parser.add_argument('--output', '-o', required=True, type=str, help='Output xml file.')
	args = parser.parse_args()
	frcmod_mol2_to_xml(args.frcmod, args.mol2, args.output)
