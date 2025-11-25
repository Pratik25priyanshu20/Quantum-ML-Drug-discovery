# Dataset Documentation

## QM9 Dataset
- **Source**: figshare.com (Ramakrishnan et al.)
- **Size**: 134,000 molecules
- **Format**: XYZ coordinates + properties
- **Properties**:
  - Dipole moment (D)
  - Isotropic polarizability (α)
  - HOMO/LUMO energies (εHOMO, εLUMO)
  - Electronic spatial extent (⟨R²⟩)
  - Zero point vibrational energy (ZPVE)
  - Internal energy at 0K, 298K (U₀, U)
  - Enthalpy at 298K (H)
  - Free energy at 298K (G)
  - Heat capacity at 298K (Cᵥ)
- **Use**: Quantum property prediction (regression)

## MoleculeNet - BACE
- **Source**: MoleculeNet (DeepChem)
- **Size**: 1,513 compounds
- **Task**: Binary classification
- **Target**: Blood-brain barrier penetration
- **Features**: SMILES strings, experimental values

## MoleculeNet - HIV
- **Source**: MoleculeNet (DeepChem)
- **Size**: 41,127 compounds
- **Task**: Binary classification
- **Target**: HIV replication inhibition
- **Features**: SMILES strings, activity labels

## MoleculeNet - Tox21
- **Source**: MoleculeNet (DeepChem)
- **Size**: 7,831 compounds
- **Task**: Multi-task classification (12 toxicity targets)
- **Features**: SMILES strings, toxicity labels

## ZINC
- **Source**: ZINC database via MOSES
- **Size**: 250,000 compounds (subset)
- **Task**: Drug-likeness, generative modeling
- **Features**: SMILES strings

## Data Splits
All datasets are split:
- Train: 80%
- Validation: 10%
- Test: 10%

Splits are stratified for classification tasks.
Random seed: 42 (for reproducibility)
