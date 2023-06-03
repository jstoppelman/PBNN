from schnetpack.transform.neighborlist import (
        APOffDiagNeighborList,
        APOffDiagPBCNeighborList)
from schnetpack.transform import CastTo32, FDSetup_OffDiag
from schnetpack.interfaces.ase_interface import AtomsConverter
import torch
import numpy as np

class Coupling:
    """
    Class used to compute the coupling between two diabatic states
    """
    def __init__(self, nn, force_indices, couplings_loc, damping=None, periodic=True, device='cuda'):
        """
        Parameters
        -----------
        nn : str
            location of the neural network used to compute the coupling term
        force_indices : list 
            Indices of atoms that the coupling NN is applied to
        couplings_loc : list
            Used in the PBNN code to add the coupling term energies and forces to the correct element of the Hamiltonian
        damping : optional, either dict or None
            Settings related to damping functions if present
        periodic : bool
            bool indicating whether periodic bundaries are being used.
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.model = torch.load(nn).to(device)
        self.force_indices = force_indices
        self.couplings_loc = couplings_loc
        self.device = device

        #Get the neighborlist transform class
        if periodic:
            neighbor_list = APOffDiagPBCNeighborList()
        else:
            neighbor_list = APNetOffDiagNeighborList()

        #Add damping function to transforms applied to Atoms object if damping is not None
        transforms = []
        if damping:
            parent_atom = damping["damping_parent"]
            dissoc_atom = damping["damping_dissoc"]
            parent_atom_prod = damping["damping_parent_product"]
            dissoc_atom_prod = damping["damping_dissoc_product"]
            
            parent_atom = parent_atom.split(',')
            parent_atom = [int(i) for i in parent_atom]
            dissoc_atom = dissoc_atom.split(',')
            dissoc_atom = [int(i) for i in dissoc_atom]

            parent_atom_prod = parent_atom_prod.split(',')
            parent_atom_prod = [int(i) for i in parent_atom_prod]
            dissoc_atom_prod = dissoc_atom_prod.split(',')
            dissoc_atom_prod = [int(i) for i in dissoc_atom_prod]

            fd_setup = FDSetup_OffDiag(parent_atom, dissoc_atom, parent_atom_prod, dissoc_atom_prod)
            transforms.append(fd_setup)
        
        transforms.append(CastTo32())
        #Converter object
        self.converter = AtomsConverter(neighbor_list=neighbor_list, transforms=transforms, device=device)

    def compute_energy_force(self, atoms, total_forces, total_potential=None):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        total_forces : np.ndarray
            numpy array containing the total intramolecular forces for a diabat
        total_potential : optional, np.ndarray or None
            Containis the electrical potential on each atom if being used for the NN

        Returns
        -----------
        energy : np.ndarray
            energy in kJ/mol
        forces : np.ndarray
            forces in kJ/mol/A
        """
        inputs = self.converter(atoms[self.force_indices])
        if total_potential is not None:
            inputs["potential_solvent"] = torch.from_numpy(total_potential).to(self.device).float()
        result = self.model(inputs)
        energy = result["y"].detach().cpu().numpy()
        forces = result["dr_y"].detach().cpu().numpy()
        forces[forces!=forces] = 0
        total_forces[self.force_indices] += forces
        return np.asarray(energy), total_forces

