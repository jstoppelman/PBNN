import torch
from schnetpack.interfaces.ase_interface import AtomsConverter
from schnetpack.transform.neighborlist import TorchNeighborList, APNetNeighborList, APNetPBCNeighborList
from schnetpack.transform import CastTo32, FDSetup, FDSetup_SchNet
import numpy as np
from ase import Atoms
import time
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import get_distances
import pickle

def shift_reacting_atom(positions, react_residue, box):
    """
    Some OpenMM forces need a residue to not be split by 
    periodic boundaries, and the "wrap" function in the 
    calculator function only wraps residues according to
    the H11 topology, which may result in spliting the 
    reacting residue across a periodic boundary. This
    function ensures that the reacting atom is wrapped
    back to the principal box with the other atoms in
    the reacting residue

    Parameters
    -----------
    positions : np.ndarray
        Positions array
    react_residue : list
        Indices of the reacting_residue
    box : object
        ASE cell object

    Returns
    ----------
    positions : np.ndarray
        Positions with the shifted atom position
    """
    react_positions = positions[react_residue]
    #Get distances (w/o minimum image) using the ASE get_distances function 
    #from atom 0 in the residue to the rest of the atoms in the residue
    D, D_len = get_distances(react_positions[0], react_positions[1:])

    D_mic, D_len_mic = get_distances(react_positions[0], react_positions[1:], cell=box, pbc=True)
    #Determine if there is a difference between the minimum image distances and the non-minimum imaged
    #distances
    diff = D - D_mic

    positions[react_residue[1:]] -= diff[0]
    return positions

def reorder(array, traj_inds):
    """
    Sometimes we want to reorder forces or position from the diabat 2 ordering to the diabat 1 ordering
    Parameters
    -----------
    forces : np.ndarray
        np array containing either forces or positions
    traj_inds : list
        list of ints containing the location of each atom from diabat 1 in diabat 2

    Returns
    -----------
    array[reord_list] : np.ndarray
        reorderd array
    """
    reord_list = [traj_inds.index(i) for i in range(len(traj_inds))]
    return array[reord_list]

class NN_Intra(Calculator):
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks, designed for single molecules intramolecular interactions
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, model, force_atoms, damping=None, device='cuda', **kwargs):
        """
        Parameters
        -----------
        model : str
            location of the neural network model for the monomer
        force_atoms : list
            List of atom indices that correspond to the atoms the neural network is applied to
        damping : list
            Indices of atoms that the damping function will be applied to
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        Calculator.__init__(self, **kwargs)

        self.model = torch.load(model).to(torch.device(device))
        self.nn_force_atoms = force_atoms
        transforms = []
        if damping:
            fd_setup = FDSetup_SchNet(damping[0], damping[1])
            transforms.append(fd_setup)
        transforms.append(CastTo32())
        neighbor_list = TorchNeighborList(8.0)
        self.converter = AtomsConverter(neighbor_list=neighbor_list, transforms=transforms, device="cuda")

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Compute the energy for the intramolecular energies

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        properties : list
            Properties that the ASE class is trying to get
        system_changes : list
            List of changes detected by the ASE class
        """
        self.results = {}
        Calculator.calculate(self, atoms)

        atoms = atoms[self.nn_force_atoms]
        inputs = self.converter(atoms)
        result = self.model(inputs)
        energy = result["y"].detach().cpu().numpy()
        forces = result["dr_y"].detach().cpu().numpy()
        forces[forces!=forces] = 0
        self.results["energy"] = energy
        self.results["forces"] = forces

class NN_Inter:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks, designed for intermolecular dimer interactions.
    """
    def __init__(self, model, res_list, ZA, ZB, damping=None, pbc=True, device='cuda'):
        """
        Parameters
        -----------
        model : str
            location of the neural network model for the dimer
        res_list : list
            List of atom indices that correspond to the atoms the neural network is applied to
        ZA : np.ndarray
            Atomic numbers of residue 1
        ZB : np.ndarray
            Atomic numbers of residue 2
        damping : list
            List of atoms that the damping function is applied to
        pbc : bool
            Whether minimum image functions need to be used
        device : str
            String indicating where the neural networks will be run. Default is cuda.
        """
        self.model = torch.load(model).to(torch.device(device))
        self.res_list = res_list
        self.nn_force_atoms = np.asarray([atom_id for res in res_list for atom_id in res])
        self.nn_force_atoms = self.nn_force_atoms.astype(int)
        nn_res_list = [[i for i in range(len(res_list[0]))]]
        monomer_2 = [i for i in range(res_list[0][-1]+1, res_list[0][-1]+1+len(res_list[1]))]
        nn_res_list.append(monomer_2)
        self.device = device
        transforms = []
        
        if damping:
            fd_setup = FDSetup(damping[0], damping[1])
            transforms.append(fd_setup)

        transforms.append(CastTo32())
        if pbc:
            neighbor_list = APNetPBCNeighborList(ZA, ZB)
        else:
            neighbor_list = APNetNeighborList(ZA, ZB)

        self.converter = AtomsConverter(neighbor_list=neighbor_list, transforms=transforms, device=self.device)

    def compute_energy_force(self, atoms, total_forces, atom_potential=None):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        atoms : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        total_forces : np.ndarray
            numpy array containing the total intramolecular forces for a diabat
        atom_potential : optional, np.ndarray
            Contains the external electric potential on each atom if the model uses this

        Returns
        -----------
        energy : np.ndarray
            Intramoleculer energy in kJ/mol
        forces : np.ndarray
            Intramolecular forces in kJ/mol/A
        """
        inputs = self.converter(atoms[self.nn_force_atoms])
        if atom_potential is not None:
            inputs["potential_solvent_A"] = torch.from_numpy(atom_potential[self.res_list[0]]).to(self.device).float()
            inputs["potential_solvent_B"] = torch.from_numpy(atom_potential[self.res_list[1]]).to(self.device).float()
        result = self.model(inputs)
        energy = result["y"].detach().cpu().numpy()
        forces = result["dr_y"].detach().cpu().numpy()
        forces[forces!=forces] = 0
        total_forces[self.nn_force_atoms] += forces

        return np.asarray(energy), total_forces

class Diabat:
    """
    Contains collection of terms that are used to model each diabat
    """
    def __init__(self, openmm, nn_intra_opts, nn_inter_opts, reorder_graph=None, shift=0):
        """
        Parameters
        -----------
        openmm_opts : dictionary
            collection of options needed to start OpenMM
        nn_intra_opts : dictionary
            collection of options need to initialize NN_Intra classes
        nn_inter_opts : dictionary
            collection of options needed to initialize NN_Inter classes
        reorder_graph : GraphReorder object
            (optional) object that reorders the diabat 1 positions to another diabat. Not needed for diabat 1
        shift : float
            (optional) shift diabat 2 to be on the same energy level as diabat 1. The electronic energy
            of the isolated monomers 
        """
        self.openmm = openmm
        if nn_intra_opts:
            self._setup_nnintra(nn_intra_opts)
        if nn_inter_opts:
            self._setup_nninter(nn_inter_opts)
        self.reorder_graph = reorder_graph
        self.shift = shift

    def _setup_nnintra(self, nnintra_opts):
        """
        Parameters
        -----------
        nnintra_opts : dictionary
            list of options needed to create the NN_Intra class
        """
        self.nnintra = []
        for opt in nnintra_opts:
            model = opt['fname']
            indices = opt["atom_index"]
            if "damping_parent" in opt.keys():
                parent_atom = opt["damping_parent"]
                dissoc_atom = opt["damping_dissoc"]
                parent_atom = parent_atom.split(',')
                parent_atom = [int(i) for i in parent_atom]
                if not isinstance(dissoc_atom, list): dissoc_atom = list(dissoc_atom)
                dissoc_atom = [int(i) for i in dissoc_atom]
                nnintra = NN_Intra(model, indices, damping=[parent_atom, dissoc_atom])
            else:
                nnintra = NN_Intra(model, indices)
            self.nnintra.append(nnintra)

    def _setup_nninter(self, nninter_opts):
        """
        Parameters
        -----------
        nnintra_opts : dictionary
            list of options needed to create the NN_Intra class
        """

        model = nninter_opts['fname']
        residue_indices = nninter_opts["indices"]
        parent_atom = nninter_opts["damping_parent"]
        dissoc_atom = nninter_opts["damping_dissoc"]
        ZA = np.asarray(nninter_opts["ZA"])
        ZB = np.asarray(nninter_opts["ZB"])
        parent_atom = parent_atom.split(',')
        parent_atom = [int(i) for i in parent_atom]
        dissoc_atom = dissoc_atom.split(',')
        dissoc_atom = [int(i) for i in dissoc_atom]

        self.nninter = NN_Inter(model, residue_indices, ZA, ZB, damping=[parent_atom, dissoc_atom])

    def compute_energy_force(self, atoms, potential=None, field=None, atom_potential=None):
        """
        Parameters
        -----------
        atoms : ASE atoms object
            atoms object
        potential : float, optional
            add potential energy from external potential if present (usually only for a training step
        field : np.ndarray, optional
            add forces from the external field if present
        atom_potential : np.ndarray, optional
            external potential on each atom

        Returns
        -----------
        energy : np.ndarray
            Contains the energy for a particular configuration
        forces : np.ndarray
            Contains the forces (ordered in diabat 1) for a particular configuration
        """
        #Determine if the Diabat object contains a GraphReorder class
        if self.reorder_graph:
            new_atoms, indices = self.reorder_graph.reorder(atoms)
        else:
            new_atoms = atoms
            indices = np.arange(len(new_atoms)).astype(int).tolist()

        if self.openmm.cutoff:
            res_list = self.openmm.res_list()
            react_residue = res_list[self.openmm.react_residue]
            positions = shift_reacting_atom(new_atoms.get_positions(), react_residue, new_atoms.get_cell())
            new_atoms.set_positions(positions)
        
        #Initialize a zeros array in order to contain all the forces
        total_forces = np.zeros_like(new_atoms.get_positions())
        
        openmm_energy, openmm_forces = self.compute_openmm_energy(new_atoms)
        
        #Loop through the NN_Intra classes and get the energies and forces
        nnintra_energy, total_forces = self.compute_nn_intra(new_atoms, total_forces)
        #Get the NN_Inter energy/force
        nninter_energy, total_forces = self.nninter.compute_energy_force(new_atoms, total_forces, atom_potential)
        
        #Combine the SAPT-FF energy/force with the neural network energy/force
        energy = openmm_energy + nnintra_energy + nninter_energy + self.shift
        forces = openmm_forces + total_forces
        
        if potential:
            energy += potential
            forces += field
       
        #Reorder with the current indices
        forces = reorder(forces, indices)
        return energy, forces

    def compute_openmm_energy(self, atoms):
        """
        Get energy from OpenMM

        Parameters
        -----------
        atoms : ASE atoms object
            atoms

        Returns 
        -----------
        oprnmm_energy : np.ndarray
            OpenMM Energy in kJ/mol
        openmm_forces : np.ndarray
            OpenMM Forces in kJ/mol/A
        """
        #Set initial positions
        self.openmm.set_initial_positions(atoms.get_positions())
        self.openmm.calculate(atoms=atoms)
        openmm_energy = self.openmm.results["energy"]
        openmm_forces = self.openmm.results["forces"]

        return openmm_energy, openmm_forces

    def compute_nn_intra(self, atoms, total_forces):
        """
        Loop through all NN Intra models and get energy and force

        Parameters
        -----------
        atoms : ASE atoms object
            atoms
        total_forces : np.ndarray
            Array for which to add all forces from the NN models

        Returns 
        -----------
        nnintra_energy : np.ndarray
            NN_Intra Energy in kJ/mol
        total_forces : np.ndarray
            Contains the forces from all the NN models
        """

        nnintra_energy = 0
        for nnintra in self.nnintra:
            nnintra.calculate(atoms=atoms)
            energy = nnintra.results["energy"]
            forces = nnintra.results["forces"]
            nnintra_energy += energy
            total_forces[nnintra.nn_force_atoms] += forces
        return nnintra_energy, total_forces

