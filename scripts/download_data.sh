#!/bin/bash
# Data Download Script for Quantum Drug Discovery
# Downloads QM9, MoleculeNet datasets (all FREE)

set -e  # Exit on error

echo "🧬 Quantum Drug Discovery - Data Download"
echo "=========================================="

# Create data directories
mkdir -p data/raw/qm9
mkdir -p data/raw/moleculenet
mkdir -p data/raw/zinc
mkdir -p data/processed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to download with progress
download_file() {
    local url=$1
    local output=$2
    local name=$3
    
    echo -e "${YELLOW}📦 Downloading ${name}...${NC}"
    
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url"
    else
        echo -e "${RED}❌ Error: wget or curl required${NC}"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${name} downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download ${name}${NC}"
        exit 1
    fi
}

# 1. Download QM9 Dataset (~500 MB)
echo ""
echo "1️⃣  QM9 Dataset (134,000 molecules)"
echo "   Source: figshare.com"

QM9_URL="https://figshare.com/ndownloader/files/3195389"
QM9_OUTPUT="data/raw/qm9/qm9.xyz"

if [ -f "$QM9_OUTPUT" ]; then
    echo -e "${GREEN}✅ QM9 already downloaded${NC}"
else
    download_file "$QM9_URL" "$QM9_OUTPUT" "QM9 Dataset"
    
    # Verify file size (should be ~500MB)
    size=$(du -h "$QM9_OUTPUT" | cut -f1)
    echo "   File size: $size"
fi

# 2. Download QM9 CSV (smaller, preprocessed version)
echo ""
echo "2️⃣  QM9 CSV (preprocessed)"

QM9_CSV_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
QM9_CSV_OUTPUT="data/raw/qm9/qm9.csv"

if [ -f "$QM9_CSV_OUTPUT" ]; then
    echo -e "${GREEN}✅ QM9 CSV already downloaded${NC}"
else
    download_file "$QM9_CSV_URL" "$QM9_CSV_OUTPUT" "QM9 CSV"
fi

# 3. Download MoleculeNet - BACE (Quantitative/Classification)
echo ""
echo "3️⃣  MoleculeNet - BACE Dataset"
echo "   (1,513 compounds for blood-brain barrier penetration)"

BACE_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
BACE_OUTPUT="data/raw/moleculenet/bace.csv"

if [ -f "$BACE_OUTPUT" ]; then
    echo -e "${GREEN}✅ BACE already downloaded${NC}"
else
    download_file "$BACE_URL" "$BACE_OUTPUT" "BACE Dataset"
fi

# 4. Download MoleculeNet - HIV (Classification)
echo ""
echo "4️⃣  MoleculeNet - HIV Dataset"
echo "   (41,127 compounds for HIV inhibition)"

HIV_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
HIV_OUTPUT="data/raw/moleculenet/hiv.csv"

if [ -f "$HIV_OUTPUT" ]; then
    echo -e "${GREEN}✅ HIV already downloaded${NC}"
else
    download_file "$HIV_URL" "$HIV_OUTPUT" "HIV Dataset"
fi

# 5. Download MoleculeNet - Tox21 (Multi-task Classification)
echo ""
echo "5️⃣  MoleculeNet - Tox21 Dataset"
echo "   (7,831 compounds for toxicity prediction)"

TOX21_URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX21_OUTPUT="data/raw/moleculenet/tox21.csv.gz"

if [ -f "$TOX21_OUTPUT" ]; then
    echo -e "${GREEN}✅ Tox21 already downloaded${NC}"
else
    download_file "$TOX21_URL" "$TOX21_OUTPUT" "Tox21 Dataset"
    
    # Decompress
    echo "   Decompressing..."
    gunzip -f "$TOX21_OUTPUT"
    echo -e "${GREEN}   ✅ Decompressed${NC}"
fi

# 6. Download ZINC subset (optional, for additional testing)
echo ""
echo "6️⃣  ZINC Dataset (250k subset)"
echo "   (Drug-like molecules)"

ZINC_URL="https://raw.githubusercontent.com/molecularsets/moses/master/data/dataset_v1.csv"
ZINC_OUTPUT="data/raw/zinc/zinc_250k.csv"

if [ -f "$ZINC_OUTPUT" ]; then
    echo -e "${GREEN}✅ ZINC already downloaded${NC}"
else
    download_file "$ZINC_URL" "$ZINC_OUTPUT" "ZINC Subset"
fi

# Create data README
echo ""
echo "7️⃣  Creating data documentation..."

cat > data/README.md << 'DATAREADME'
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
DATAREADME

echo -e "${GREEN}✅ Data documentation created${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✅ DATA DOWNLOAD COMPLETE${NC}"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "   • QM9: 134,000 molecules (~500 MB)"
echo "   • BACE: 1,513 compounds"
echo "   • HIV: 41,127 compounds"
echo "   • Tox21: 7,831 compounds"
echo "   • ZINC: 250,000 molecules"
echo ""
echo "📁 Data location: data/raw/"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: python src/data/loaders.py  (verify data)"
echo "   2. Open: notebooks/01_data_exploration.ipynb"
echo ""