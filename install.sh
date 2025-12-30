#!/bin/bash
# Installation script for COULOMB_MC
# Run: bash install.sh

set -e  # Exit on error

echo "========================================================================"
echo "COULOMB_MC Installation Script"
echo "========================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo ""
echo "1. Creating conda environment 'coulomb_mc'..."
conda create -n coulomb_mc python=3.11 numpy scipy matplotlib numba jupyter pytest -y

# Activate environment
echo ""
echo "2. Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate coulomb_mc

# Install package
echo ""
echo "3. Installing coulomb_mc package..."
pip install -e .

# Copy NIST data
echo ""
echo "4. Copying NIST data files..."
mkdir -p data/nist

if [ -f "COULOMB_LET/STOPRANGE_NIST_Proton.DAT" ]; then
    cp COULOMB_LET/STOPRANGE_NIST_Proton.DAT data/nist/
    echo "   ✓ Copied STOPRANGE_NIST_Proton.DAT"
else
    echo "   ⚠ STOPRANGE_NIST_Proton.DAT not found in COULOMB_LET/"
fi

if [ -f "COULOMB_LET/STOPRANGE_NIST_Helium.DAT" ]; then
    cp COULOMB_LET/STOPRANGE_NIST_Helium.DAT data/nist/
    echo "   ✓ Copied STOPRANGE_NIST_Helium.DAT"
else
    echo "   ⚠ STOPRANGE_NIST_Helium.DAT not found in COULOMB_LET/"
fi

if [ -f "COULOMB_LET/Pstar_NIST_Water.dat" ]; then
    cp COULOMB_LET/Pstar_NIST_Water.dat data/nist/
    echo "   ✓ Copied Pstar_NIST_Water.dat"
else
    echo "   ⚠ Pstar_NIST_Water.dat not found in COULOMB_LET/"
fi

# Create __init__ files
echo ""
echo "5. Creating Python package files..."
for dir in coulomb_mc/transport coulomb_mc/scoring coulomb_mc/ml coulomb_mc/io; do
    touch $dir/__init__.py
done

# Test installation
echo ""
echo "6. Testing installation..."
python examples/scripts/test_installation.py

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""
echo "To activate the environment:"
echo "  conda activate coulomb_mc"
echo ""
echo "To test:"
echo "  python examples/scripts/test_installation.py"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
