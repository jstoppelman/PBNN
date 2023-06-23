import numpy as np

class TestComponent:
    """
    Test Component of the PBNN Hamiltonian,
    like NNIntra against a dataset
    """
    def __init__(self, data, energy, forces, calculator, name):
        """
        Parameters
        -----------
        data : list
            List of ASE atoms objects
        energy : list
            List of energies for each structure in the dataset (if energy property is present)
        forces : list
            List of forces for each structure in the dataset
        calculator : ASE Calculator object
            Calculator used for computing energies and forces
        name : str
            Output file name
        """
        self.data = data
        self.energy = energy
        self.forces = forces
        self.calc = calculator
        self.name = name

    def run_test(self):
        """
        Compute Calculator energies and forces and save npy arrays
        """
        test_energy = []
        test_forces = []
        for i, frame in enumerate(self.data):
            print(f"Structure {i+1}/{len(self.data)}")
            frame.calc = self.calc
            energy = frame.get_potential_energy()
            test_energy.append(energy)
            forces = frame.get_forces()
            test_forces.append(forces)

        np.save(f"ref_energy_{self.name}.npy", self.energy)
        np.save(f"test_energy_{self.name}.npy", np.asarray(test_energy))
        np.save(f"ref_forces_{self.name}.npy", self.forces)
        np.save(f"test_forces_{self.name}.npy", np.asarray(test_forces))
