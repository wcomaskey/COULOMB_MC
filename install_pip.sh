#!/bin/bash
# Installation script for COULOMB_MC using pip (no conda required)
# Run: bash install_pip.sh

set -e  # Exit on error

echo "========================================================================"
echo "COULOMB_MC Installation Script (pip version)"
echo "========================================================================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo ""
echo "1. Using Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
python3 -m venv venv_coulomb_mc

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv_coulomb_mc/bin/activate

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "5. Installing Python dependencies..."
pip install numpy scipy matplotlib numba jupyter pytest

# Install package in editable mode
echo ""
echo "6. Installing coulomb_mc package..."
pip install -e .

# Copy NIST data
echo ""
echo "7. Copying NIST data files..."
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
echo "8. Creating Python package files..."
for dir in coulomb_mc/transport coulomb_mc/scoring coulomb_mc/ml coulomb_mc/io; do
    mkdir -p $dir
    touch $dir/__init__.py
done

# Test installation
echo ""
echo "9. Testing installation..."
python examples/scripts/test_installation.py

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""
echo "To activate the environment:"
echo "  source venv_coulomb_mc/bin/activate"
echo ""
echo "To test Week 2 features:"
echo "  python examples/scripts/test_week2.py"
echo ""
echo "To run Bragg peak simulation:"
echo "  python examples/scripts/bragg_peak_simple.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
