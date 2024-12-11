from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from collections import Counter

def smiles_properties(smiles): # returns the wanted molecular informations from the SMILES
    mol = Chem.MolFromSmiles(smiles)  
    if mol:  
        # molecular weight
        weight = Descriptors.MolWt(mol)
        # number of hydrogenes donnors or acceptors of electrons
        donnorsH = Descriptors.NumHDonors(mol)
        acceptorsH = Descriptors.NumHAcceptors(mol)  
        # number of rings
        rings = mol.GetRingInfo().NumRings()
        # number of each atom in the molecule
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()] 
        atoms_count = dict(Counter(atoms))
        # number of single, double and triple bonds
        single_bonds = 0
        double_bonds = 0
        triple_bonds = 0
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                single_bonds += 1
            elif bond_type == Chem.BondType.DOUBLE:
                double_bonds += 1
            elif bond_type == Chem.BondType.TRIPLE:
                triple_bonds += 1
        # partial charges of each atom of the molecule
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp('_GasteigerCharge') # takes the partial charge of the atom
            atom_symbol = atom.GetSymbol()  
            charges.append((atom_symbol, charge))
        # how ramificated the molecule is
        ramification = Descriptors.HallKierAlpha(mol)
        # number of valence electrons 
        valence_electrons = Descriptors.NumValenceElectrons(mol)

        return weight, donnorsH, acceptorsH, rings, atoms_count, single_bonds, double_bonds, triple_bonds, charges, ramification, valence_electrons

    else:
        return None 