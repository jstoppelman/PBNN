import plumed
import numpy as np

class PlumedInterface:

    def __init__(self, input_string, num_atoms, log):
        """
        Plumed interface is used for implementing enhanced sampling methods in the QM/MM simulations

        Parameters
        ----------
        input string: str
            Single string where each line of the Plumed input file is separated by a new line
            character. Determines the flavor of enhanced sampling being performed and the
            atoms to which it is applied.

        num_atoms: int
            Total number of atoms for which forces are computed

        log: str
            Name of file to which Plumed will write
        """
        self.input_string = input_string
        self.num_atoms = num_atoms
        self.log = log

        self.plumed = plumed.Plumed()
        self.plumed.cmd('setMDEngine', 'python')
        self.plumed.cmd("setNatoms", self.num_atoms)
        self.plumed.cmd("setMDLengthUnits", 1/10)
        self.plumed.cmd("setMDTimeUnits", 1/1000)
        self.plumed.cmd("setMDMassUnits", 1.)
        self.plumed.cmd('setTimestep', 1.)
        self.plumed.cmd("setKbT", 1.)
        self.plumed.cmd("setLogFile", self.log)
        self.plumed.cmd("init")

        for line in self.input_string.split('\n'):
            self.plumed.cmd("readInputLine", line)

    def compute_bias(self, positions, step, unbiased_energy, masses, unit_cell):
        """
        Computed the biasing forces and energy for a particular enhanced sampling method given the following parameters...

        Parameters
        ----------
        positions:
            Numpy array of Cartesian coordinates for all atoms which forces are applied to

        step: int
            Current frame/timestep

        unbiased_energy: float
            Total system energy (OpenMM + Psi4) prior to adding bias

        masses: Numpy array
            Array of masses for each atom. Set via atoms.get_masses() !!! Needs to be replaced when ASE is removed

        unit_cell: Numpy array
             Array of unit cell vectors. Set via atoms.get_cell() !!! Needs to be replaced when ASE is removed
        """
        self.plumed.cmd("setStep", step)

        self.plumed.cmd("setBox", unit_cell)

        self.plumed.cmd("setPositions", positions)
        self.plumed.cmd("setEnergy", unbiased_energy)
        self.plumed.cmd("setMasses", masses)
        forces_bias = np.zeros(positions.shape)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)
        return [energy_bias, forces_bias]

    def finalize(self):
        self.plumed.finalize()



