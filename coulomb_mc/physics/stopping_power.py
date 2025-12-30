"""
Stopping power calculations using NIST data tables.

Ported from GCR_Modern.f90 (Proton_Properties and Helium_Properties modules)
"""

import numpy as np
import numba
from pathlib import Path
from typing import Tuple, Optional

class StoppingPower:
    """
    Stopping power and range calculations for ions in matter.

    Uses NIST PSTAR/ASTAR data for protons and alpha particles,
    with Barkas effective charge scaling for heavy ions.

    References:
        - NIST PSTAR: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
        - GCR_Modern.f90 lines 868-1026 (Proton_Properties module)
    """

    # Material ID mapping (compatible with NIST tables)
    MATERIALS = {
        'water': 0,
        'aluminum': 1,
        'polyethylene': 2,
        'muscle': 3,
        'hydrogen': 4,
        'graphite': 5,
        'silicon': 6,
        'iron': 7,
        'air': 8,
        'co2': 9,
    }

    def __init__(self, material: str = 'water', data_dir: Optional[Path] = None):
        """
        Initialize stopping power calculator.

        Parameters:
            material: Material name (see MATERIALS dict)
            data_dir: Directory containing NIST data files (auto-detected if None)
        """
        self.material = material.lower()

        if self.material not in self.MATERIALS:
            raise ValueError(f"Unknown material '{material}'. "
                           f"Available: {list(self.MATERIALS.keys())}")

        self.material_id = self.MATERIALS[self.material]

        # Auto-detect data directory
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'nist'
        self.data_dir = Path(data_dir)

        # Storage for NIST data
        self.energy = None  # MeV/u
        self.stopping_power = None  # MeV cm²/g (10 materials)
        self.range_data = None  # g/cm²

        # Load data
        self._load_nist_data()

    def _load_nist_data(self):
        """
        Load NIST stopping power and range tables.

        Tries binary .npy format first (100x faster), falls back to ASCII .DAT.

        File format:
            Line 1: ND (number of data points)
            Line 2-3: Labels
            Lines 4+: E, S1, R1, S2, R2, ..., S10, R10
        """
        npy_file = self.data_dir / 'STOPRANGE_NIST_Proton.npy'
        dat_file = self.data_dir / 'STOPRANGE_NIST_Proton.DAT'

        # Try binary first (fast)
        if npy_file.exists():
            data = np.load(npy_file)
            self.energy = data[:, 0]
            # Interleaved: S1, R1, S2, R2, ... S10, R10
            self.stopping_power = data[:, 1::2].T  # Every odd column, transposed
            self.range_data = data[:, 2::2].T      # Every even column, transposed
            return

        # Fall back to ASCII
        if not dat_file.exists():
            raise FileNotFoundError(
                f"NIST data file not found: {dat_file}\n"
                f"Please copy from COULOMB_LET/STOPRANGE_NIST_Proton.DAT"
            )

        filename = dat_file

        with open(filename, 'r') as f:
            # Read number of data points
            nd = int(f.readline().strip())

            # Read labels
            label1 = f.readline()
            label2 = f.readline()

            # Allocate arrays
            self.energy = np.zeros(nd)
            self.stopping_power = np.zeros((10, nd))  # 10 materials
            self.range_data = np.zeros((10, nd))

            # Read data
            for i in range(nd):
                line = f.readline().split()
                self.energy[i] = float(line[0])

                # Read S1, R1, S2, R2, ..., S10, R10
                for j in range(10):
                    self.stopping_power[j, i] = float(line[1 + 2*j])
                    self.range_data[j, i] = float(line[2 + 2*j])

        print(f"Loaded NIST data: {nd} energy points for {filename.name}")

        # Auto-convert to binary for faster loading next time
        if not npy_file.exists():
            # Reconstruct full data array (interleaved format)
            data = np.zeros((nd, 21))  # Energy + 10*(S,R) = 21 columns
            data[:, 0] = self.energy
            for j in range(10):
                data[:, 1 + 2*j] = self.stopping_power[j, :]
                data[:, 2 + 2*j] = self.range_data[j, :]
            np.save(npy_file, data)
            print(f"Auto-converted to binary: {npy_file.name} (faster next time)")

    @staticmethod
    @numba.njit(fastmath=True, cache=True)
    def _phibin_interpolate(x_array: np.ndarray, y_array: np.ndarray,
                            x: float) -> float:
        """
        Binary search + power-law interpolation.

        Ported from GCR_Modern.f90:367-388 (phibin function)

        Uses log-log power-law interpolation:
            y(x) = y1 * (x/x1)^a
        where:
            a = log(y2/y1) / log(x2/x1)

        Parameters:
            x_array: Sorted array of x values
            y_array: Corresponding y values
            x: Point to interpolate

        Returns:
            Interpolated y value
        """
        n = len(x_array)

        # Binary search for interval
        ir = np.searchsorted(x_array, x)

        # Bounds checking
        if ir <= 0:
            return y_array[0]
        elif ir >= n:
            return y_array[-1]

        # Power-law interpolation
        if y_array[ir] * y_array[ir-1] > 0:
            a = np.log(y_array[ir] / y_array[ir-1]) / \
                np.log(x_array[ir] / x_array[ir-1])
            result = y_array[ir] * (x / x_array[ir])**a
        else:
            result = y_array[ir]

        return result

    def stopping_power_MeV_cm2_g(self, energy_MeV: float, A: float,
                                   Z: float) -> float:
        """
        Calculate stopping power in MeV cm²/g.

        Ported from GCR_Modern.f90:972-992 (SMAT function)

        Uses Barkas effective charge formula for Z > 1.

        Parameters:
            energy_MeV: Kinetic energy per nucleon [MeV/u]
            A: Atomic mass [u]
            Z: Atomic number

        Returns:
            Stopping power [MeV cm²/g]
        """
        # Helium correction for E < 250 MeV (if needed)
        if Z == 2 and energy_MeV <= 250.0:
            # TODO: Load helium-specific data for better accuracy
            pass

        # Interpolate base stopping power from proton data
        sp = self._phibin_interpolate(
            self.energy,
            self.stopping_power[self.material_id, :],
            energy_MeV
        )

        # Effective charge correction
        z_eff = self.effective_charge(energy_MeV, Z)

        return sp * z_eff**2

    def range_g_cm2(self, energy_MeV: float, A: float, Z: float) -> float:
        """
        Calculate range in g/cm².

        Ported from GCR_Modern.f90:935-960 (RMAT function)

        Parameters:
            energy_MeV: Kinetic energy per nucleon [MeV/u]
            A: Atomic mass [u]
            Z: Atomic number

        Returns:
            Range [g/cm²]
        """
        if energy_MeV <= 0:
            return 0.0

        # Interpolate base range from proton data
        range_base = self._phibin_interpolate(
            self.energy,
            self.range_data[self.material_id, :],
            energy_MeV
        )

        # Scale by A/Z²
        return range_base * A / Z**2

    @staticmethod
    @numba.njit(fastmath=True, cache=True)
    def effective_charge(energy_MeV: float, Z: float) -> float:
        """
        Calculate effective charge using Barkas formula.

        Ported from GCR_Modern.f90:913-922 (zeffz function)

        The effective charge accounts for electron capture/loss
        at low velocities. Formula from Barkas (1963).

        Parameters:
            energy_MeV: Kinetic energy per nucleon [MeV/u]
            Z: Atomic number

        Returns:
            Effective charge Z_eff (≤ Z)
        """
        # Lorentz factor
        proton_mass = 938.0  # MeV/c²
        gamma = 1.0 + energy_MeV / proton_mass

        # Velocity (β = v/c)
        beta = np.sqrt(1.0 - 1.0 / gamma**2)

        # Barkas effective charge
        if Z > 1:
            z_eff = Z * (1.0 - np.exp(-125.0 * beta / Z**(2.0/3.0)))
        else:
            z_eff = Z

        return z_eff

    def LET_keV_um(self, energy_MeV: float, A: float, Z: float) -> float:
        """
        Calculate LET (Linear Energy Transfer) in keV/µm.

        This is the most commonly used unit in radiobiology.

        Parameters:
            energy_MeV: Kinetic energy per nucleon [MeV/u]
            A: Atomic mass [u]
            Z: Atomic number

        Returns:
            LET [keV/µm]
        """
        # Stopping power in MeV cm²/g
        sp = self.stopping_power_MeV_cm2_g(energy_MeV, A, Z)

        # Convert to keV/µm
        # 1 MeV cm²/g = 0.1 keV/µm for ρ=1 g/cm³
        # For general density: multiply by ρ
        # Here we return mass stopping power / 10
        return sp / 10.0

    @staticmethod
    def _parse_particle_type(particle_type: str) -> Tuple[float, float]:
        """
        Parse particle string to (A, Z).

        Examples:
            'proton' or 'H-1' → (1, 1)
            'C-12' → (12, 6)
            'alpha' or 'He-4' → (4, 2)
        """
        particles = {
            'proton': (1, 1),
            'H-1': (1, 1),
            'deuteron': (2, 1),
            'H-2': (2, 1),
            'triton': (3, 1),
            'H-3': (3, 1),
            'He-3': (3, 2),
            'He-4': (4, 2),
            'alpha': (4, 2),
            'Li-7': (7, 3),
            'Be-9': (9, 4),
            'B-11': (11, 5),
            'C-12': (12, 6),
            'N-14': (14, 7),
            'O-16': (16, 8),
            'Ne-20': (20, 10),
            'Si-28': (28, 14),
            'Fe-56': (56, 26),
        }
        return particles.get(particle_type, (12, 6))  # Default C-12


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize
    sp = StoppingPower(material='water')

    # Test with Carbon-12 at 400 MeV/u in water
    energy = 400.0  # MeV/u
    A = 12.0
    Z = 6.0

    let = sp.LET_keV_um(energy, A, Z)
    range_gcm2 = sp.range_g_cm2(energy, A, Z)

    print(f"\nCarbon-12 @ {energy} MeV/u in water:")
    print(f"  LET: {let:.2f} keV/µm")
    print(f"  Range: {range_gcm2:.2f} g/cm²")
    print(f"  Range: {range_gcm2:.2f} cm (ρ=1 g/cm³)")
    print(f"\nExpected (ICRU Report 73):")
    print(f"  LET: ~11.2 keV/µm")
    print(f"  Range: ~26.4 cm in water")

    # Test effective charge scaling
    print(f"\nEffective charge vs energy:")
    for E in [10, 100, 1000]:
        z_eff = sp.effective_charge(E, Z)
        print(f"  {E:4.0f} MeV/u: Z_eff = {z_eff:.3f} (Z = {Z:.0f})")
