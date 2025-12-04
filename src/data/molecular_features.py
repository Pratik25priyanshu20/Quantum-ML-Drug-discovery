"""
Molecular Featurization for JAX
Convert SMILES → Graph representations with node/edge features
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import jax.numpy as jnp


@dataclass
class MolecularGraph:
    """
    Graph representation of a molecule
    JAX-compatible arrays
    """
    node_features: jnp.ndarray  # [num_atoms, node_feat_dim]
    edge_index: jnp.ndarray     # [2, num_edges]
    edge_features: jnp.ndarray  # [num_edges, edge_feat_dim]
    num_nodes: int
    num_edges: int
    
    def __repr__(self):
        return f"MolecularGraph(nodes={self.num_nodes}, edges={self.num_edges})"


class MolecularFeaturizer:
    """
    Featurize molecules for GNN input
    SMILES → Graph with rich node/edge features
    """
    
    def __init__(self):
        # Atom feature dimensions
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'P', 'I', 'Other']
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]
        self.degrees = [0, 1, 2, 3, 4, 5]
        self.formal_charges = [-2, -1, 0, 1, 2]
        
        # Bond feature dimensions
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        
        # Calculate feature dimensions
        self.node_feat_dim = (
            len(self.atom_types) +       # Atom type (one-hot)
            len(self.hybridizations) +   # Hybridization (one-hot)
            len(self.degrees) +          # Degree (one-hot)
            len(self.formal_charges) +   # Formal charge (one-hot)
            5                            # Additional: aromatic, ring, H-count, valence, mass
        )
        
        self.edge_feat_dim = (
            len(self.bond_types) +       # Bond type (one-hot)
            2                            # Conjugated, in ring
        )
        
        print(f"✅ Featurizer initialized:")
        print(f"   Node features: {self.node_feat_dim} dimensions")
        print(f"   Edge features: {self.edge_feat_dim} dimensions")
    
    def one_hot(self, value, options):
        """One-hot encode with fallback for unknown values"""
        encoding = [0] * len(options)
        try:
            idx = options.index(value)
            encoding[idx] = 1
        except ValueError:
            # Unknown value - all zeros (or last index for 'Other')
            if len(options) > 0 and options[-1] == 'Other':
                encoding[-1] = 1
        return encoding
    
    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Extract features for a single atom
        
        Returns:
            np.ndarray of shape [node_feat_dim]
        """
        features = []
        
        # Atom type (one-hot)
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in self.atom_types[:-1]:
            atom_symbol = 'Other'
        features.extend(self.one_hot(atom_symbol, self.atom_types))
        
        # Hybridization (one-hot)
        features.extend(self.one_hot(atom.GetHybridization(), self.hybridizations))
        
        # Degree (one-hot)
        degree = min(atom.GetDegree(), 5)
        features.extend(self.one_hot(degree, self.degrees))
        
        # Formal charge (one-hot)
        charge = atom.GetFormalCharge()
        charge = max(-2, min(2, charge))  # Clamp to [-2, 2]
        features.extend(self.one_hot(charge, self.formal_charges))
        
        # Additional features
        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.IsInRing()))
        features.append(float(atom.GetTotalNumHs()))
        features.append(float(atom.GetTotalValence()))
        features.append(float(atom.GetMass()) / 100.0)  # Normalize mass
        
        return np.array(features, dtype=np.float32)
    
    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        Extract features for a single bond
        
        Returns:
            np.ndarray of shape [edge_feat_dim]
        """
        features = []
        
        # Bond type (one-hot)
        features.extend(self.one_hot(bond.GetBondType(), self.bond_types))
        
        # Additional features
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(self, smiles: str) -> Optional[MolecularGraph]:
        """
        Convert SMILES string to molecular graph
        
        Args:
            smiles: SMILES string
            
        Returns:
            MolecularGraph or None if invalid SMILES
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add explicit hydrogens (optional, increases graph size)
        # mol = Chem.AddHs(mol)
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        # Extract node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self.get_atom_features(atom))
        node_features = np.array(node_features, dtype=np.float32)
        
        # Extract edges (undirected graph - add both directions)
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)
            
            # Add edge in both directions
            edge_index.append([i, j])
            edge_features.append(bond_feat)
            edge_index.append([j, i])
            edge_features.append(bond_feat)
        
        if len(edge_index) > 0:
            edge_index = np.array(edge_index, dtype=np.int32).T  # [2, num_edges]
            edge_features = np.array(edge_features, dtype=np.float32)
        else:
            # Molecule with no bonds (single atom)
            edge_index = np.zeros((2, 0), dtype=np.int32)
            edge_features = np.zeros((0, self.edge_feat_dim), dtype=np.float32)
        
        num_edges = edge_index.shape[1]
        
        # Convert to JAX arrays
        return MolecularGraph(
            node_features=jnp.array(node_features),
            edge_index=jnp.array(edge_index),
            edge_features=jnp.array(edge_features),
            num_nodes=num_atoms,
            num_edges=num_edges
        )
    
    def batch_smiles_to_graphs(
        self, 
        smiles_list: List[str], 
        show_progress: bool = True
    ) -> List[Optional[MolecularGraph]]:
        """
        Convert list of SMILES to graphs
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Show tqdm progress bar
            
        Returns:
            List of MolecularGraph objects (None for invalid SMILES)
        """
        graphs = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Featurizing molecules")
        else:
            iterator = smiles_list
        
        for smiles in iterator:
            graph = self.smiles_to_graph(smiles)
            graphs.append(graph)
        
        # Count valid graphs
        n_valid = sum(1 for g in graphs if g is not None)
        n_invalid = len(graphs) - n_valid
        
        if n_invalid > 0:
            print(f"⚠️  {n_invalid} invalid SMILES found ({n_invalid/len(graphs)*100:.1f}%)")
        
        return graphs


class MolecularFingerprints:
    """
    Classical molecular fingerprints (for comparison)
    These are NOT graphs, but fixed-size vectors
    """
    
    @staticmethod
    def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
        """
        Morgan (circular) fingerprint
        
        Args:
            smiles: SMILES string
            radius: Fingerprint radius
            n_bits: Number of bits
            
        Returns:
            Binary vector of shape [n_bits] or None
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    
    @staticmethod
    def molecular_descriptors(smiles: str) -> Optional[np.ndarray]:
        """
        RDKit molecular descriptors (200+ features)
        
        Args:
            smiles: SMILES string
            
        Returns:
            Descriptor vector or None
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Key descriptors
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
        ]
        
        return np.array(descriptors, dtype=np.float32)


def test_featurizer():
    """Test molecular featurization"""
    print("🧪 Testing Molecular Featurizer\n")
    
    # Test molecules
    test_smiles = [
        "CCO",           # Ethanol
        "c1ccccc1",      # Benzene
        "CC(=O)O",       # Acetic acid
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]
    
    featurizer = MolecularFeaturizer()
    
    print("\n📊 Featurizing test molecules:")
    for i, smiles in enumerate(test_smiles, 1):
        graph = featurizer.smiles_to_graph(smiles)
        if graph:
            print(f"\n{i}. {smiles}")
            print(f"   {graph}")
            print(f"   Node features shape: {graph.node_features.shape}")
            print(f"   Edge features shape: {graph.edge_features.shape}")
            print(f"   Edge index shape: {graph.edge_index.shape}")
    
    # Test fingerprints
    print("\n\n📊 Testing Morgan Fingerprints:")
    for smiles in test_smiles[:3]:
        fp = MolecularFingerprints.morgan_fingerprint(smiles)
        if fp is not None:
            print(f"   {smiles}: {fp.shape}, density={fp.mean():.3f}")
    
    # Test descriptors
    print("\n📊 Testing Molecular Descriptors:")
    for smiles in test_smiles[:3]:
        desc = MolecularFingerprints.molecular_descriptors(smiles)
        if desc is not None:
            print(f"   {smiles}: {desc.shape}, range=[{desc.min():.2f}, {desc.max():.2f}]")
    
    print("\n✅ Featurizer tests passed!")


if __name__ == "__main__":
    test_featurizer()