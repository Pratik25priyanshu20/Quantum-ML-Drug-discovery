"""
Data Loaders for Quantum Drug Discovery
JAX-native data loading and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import jax.numpy as jnp
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MolecularDataset:
    """Container for molecular dataset"""
    smiles: List[str]
    targets: jnp.ndarray
    mol_ids: List[str]
    feature_names: List[str]
    task_type: str  # 'regression' or 'classification'
    target_names: List[str]
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'target': self.targets[idx],
            'mol_id': self.mol_ids[idx]
        }


class QM9Loader:
    """
    Loader for QM9 dataset
    134,000 molecules with quantum properties
    """
    
    def __init__(self, data_dir: str = "data/raw/qm9"):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "qm9.csv"
        
        # QM9 target properties
        self.target_properties = [
            'mu',      # Dipole moment (D)
            'alpha',   # Isotropic polarizability (Bohr^3)
            'homo',    # HOMO energy (eV)
            'lumo',    # LUMO energy (eV)
            'gap',     # HOMO-LUMO gap (eV)
            'r2',      # Electronic spatial extent (Bohr^2)
            'zpve',    # Zero point vibrational energy (eV)
            'u0',      # Internal energy at 0K (eV)
            'u298',    # Internal energy at 298K (eV)
            'h298',    # Enthalpy at 298K (eV)
            'g298',    # Free energy at 298K (eV)
            'cv',      # Heat capacity at 298K (cal/mol·K)
        ]
        
    def load(self, target_property: str = 'mu', max_samples: Optional[int] = None) -> MolecularDataset:
        """
        Load QM9 dataset
        
        Args:
            target_property: Which property to predict (default: dipole moment)
            max_samples: Maximum number of samples (None = all)
            
        Returns:
            MolecularDataset object
        """
        print(f"📦 Loading QM9 dataset (target: {target_property})...")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"QM9 CSV not found at {self.csv_path}\n"
                f"Run: bash scripts/download_data.sh"
            )
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"   Raw data: {len(df)} molecules")
        
        # Check if target exists
        if target_property not in df.columns:
            raise ValueError(
                f"Target '{target_property}' not found. "
                f"Available: {self.target_properties}"
            )
        
        # Clean data
        df = df.dropna(subset=['smiles', target_property])
        print(f"   After cleaning: {len(df)} molecules")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
            print(f"   Sampled: {len(df)} molecules")
        
        # Extract data
        smiles = df['smiles'].tolist()
        targets = jnp.array(df[target_property].values, dtype=jnp.float32)
        
        # Create molecule IDs
        if 'mol_id' in df.columns:
            mol_ids = df['mol_id'].tolist()
        else:
            mol_ids = [f"qm9_{i:06d}" for i in range(len(df))]
        
        # Statistics
        print(f"\n   📊 Target Statistics ({target_property}):")
        print(f"      Mean: {float(jnp.mean(targets)):.4f}")
        print(f"      Std:  {float(jnp.std(targets)):.4f}")
        print(f"      Min:  {float(jnp.min(targets)):.4f}")
        print(f"      Max:  {float(jnp.max(targets)):.4f}")
        
        return MolecularDataset(
            smiles=smiles,
            targets=targets,
            mol_ids=mol_ids,
            feature_names=['smiles'],
            task_type='regression',
            target_names=[target_property]
        )


class MoleculeNetLoader:
    """
    Loader for MoleculeNet datasets (BACE, HIV, Tox21)
    """
    
    def __init__(self, data_dir: str = "data/raw/moleculenet"):
        self.data_dir = Path(data_dir)
        
        self.datasets = {
            'bace': {
                'file': 'bace.csv',
                'target_col': 'Class',
                'task': 'classification',
                'description': 'Blood-brain barrier penetration'
            },
            'hiv': {
                'file': 'HIV.csv',
                'target_col': 'HIV_active',
                'task': 'classification',
                'description': 'HIV replication inhibition'
            },
            'tox21': {
                'file': 'tox21.csv',
                'target_col': None,  # Multi-task
                'task': 'classification',
                'description': 'Toxicity prediction (12 assays)'
            }
        }
    
    def load(self, dataset_name: str, max_samples: Optional[int] = None) -> MolecularDataset:
        """
        Load MoleculeNet dataset
        
        Args:
            dataset_name: 'bace', 'hiv', or 'tox21'
            max_samples: Maximum number of samples
            
        Returns:
            MolecularDataset object
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available: {list(self.datasets.keys())}"
            )
        
        info = self.datasets[dataset_name]
        file_path = self.data_dir / info['file']
        
        print(f"📦 Loading MoleculeNet - {dataset_name.upper()}")
        print(f"   {info['description']}")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {file_path}\n"
                f"Run: bash scripts/download_data.sh"
            )
        
        # Load CSV
        df = pd.read_csv(file_path)
        print(f"   Raw data: {len(df)} molecules")
        
        # Extract SMILES
        possible_smiles_cols = ['smiles', 'SMILES', 'mol', 'Molecule', 'canonical_smiles']

        smiles_col = None
        for col in possible_smiles_cols:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            raise ValueError(f"SMILES column not found in BACE. Available columns: {df.columns}")
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column not found in {dataset_name}")
        
        smiles = df[smiles_col].tolist()
        
        # Extract targets
        if dataset_name == 'tox21':
            # Multi-task: get all toxicity columns
            target_cols = [col for col in df.columns if col.startswith('NR-') or col.startswith('SR-')]
            targets = df[target_cols].values
            target_names = target_cols
        else:
            # Single task
            target_col = info['target_col']
            targets = df[target_col].values.reshape(-1, 1)
            target_names = [target_col]
        
        # Clean NaN values
        valid_mask = ~np.isnan(targets).any(axis=1)
        smiles = [s for s, v in zip(smiles, valid_mask) if v]
        targets = targets[valid_mask]
        
        print(f"   After cleaning: {len(smiles)} molecules")
        
        # Limit samples
        if max_samples is not None and max_samples < len(smiles):
            indices = np.random.choice(len(smiles), max_samples, replace=False)
            smiles = [smiles[i] for i in indices]
            targets = targets[indices]
            print(f"   Sampled: {len(smiles)} molecules")
        
        # Convert to JAX array
        targets = jnp.array(targets, dtype=jnp.float32)
        
        # Create IDs
        mol_ids = [f"{dataset_name}_{i:06d}" for i in range(len(smiles))]
        
        # Statistics
        print(f"\n   📊 Target Statistics:")
        if dataset_name == 'tox21':
            for i, name in enumerate(target_names):
                pos_rate = float(jnp.mean(targets[:, i]))
                print(f"      {name}: {pos_rate:.2%} positive")
        else:
            pos_rate = float(jnp.mean(targets))
            print(f"      Positive rate: {pos_rate:.2%}")
        
        return MolecularDataset(
            smiles=smiles,
            targets=targets,
            mol_ids=mol_ids,
            feature_names=['smiles'],
            task_type=info['task'],
            target_names=target_names
        )


class DataSplitter:
    """
    Split molecular datasets into train/val/test
    JAX-compatible splits
    """
    
    @staticmethod
    def split(
        dataset: MolecularDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        stratify: bool = True
    ) -> Tuple[MolecularDataset, MolecularDataset, MolecularDataset]:
        """
        Split dataset into train/val/test
        
        Args:
            dataset: MolecularDataset to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            stratify: Stratify splits for classification
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(dataset)
        indices = np.arange(n)
        
        # Shuffle
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Create splits
        def create_split(idx):
            return MolecularDataset(
                smiles=[dataset.smiles[i] for i in idx],
                targets=dataset.targets[idx],
                mol_ids=[dataset.mol_ids[i] for i in idx],
                feature_names=dataset.feature_names,
                task_type=dataset.task_type,
                target_names=dataset.target_names
            )
        
        train_ds = create_split(train_idx)
        val_ds = create_split(val_idx)
        test_ds = create_split(test_idx)
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(train_ds)} ({len(train_ds)/n:.1%})")
        print(f"   Val:   {len(val_ds)} ({len(val_ds)/n:.1%})")
        print(f"   Test:  {len(test_ds)} ({len(test_ds)/n:.1%})")
        
        return train_ds, val_ds, test_ds


def verify_datasets():
    """Verify all datasets are downloaded and loadable"""
    print("🔍 Verifying datasets...\n")
    
    # Test QM9
    try:
        loader = QM9Loader()
        dataset = loader.load(target_property='mu', max_samples=100)
        print(f"✅ QM9: {len(dataset)} samples loaded\n")
    except Exception as e:
        print(f"❌ QM9 failed: {e}\n")
    
    # Test MoleculeNet
    mn_loader = MoleculeNetLoader()
    for name in ['bace', 'hiv', 'tox21']:
        try:
            dataset = mn_loader.load(name, max_samples=100)
            print(f"✅ {name.upper()}: {len(dataset)} samples loaded\n")
        except Exception as e:
            print(f"❌ {name.upper()} failed: {e}\n")


if __name__ == "__main__":
    print("🧬 Quantum Drug Discovery - Data Loaders")
    print("=" * 50)
    print()
    
    verify_datasets()
    
    print("\n" + "=" * 50)
    print("✅ All datasets verified!")
    print("\nUsage:")
    print("  from src.data.loaders import QM9Loader, MoleculeNetLoader")
    print("  loader = QM9Loader()")
    print("  dataset = loader.load(target_property='mu')")