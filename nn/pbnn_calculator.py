from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import sys, os, shutil, time
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
import random as rd
from ase import Atoms, units
from ase.io import read, write
from ase.io import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from .plumed_calculator import Plumed
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.geometry import find_mic, get_distances
from MDAnalysis import *
from MDAnalysis.analysis import distances
from MDAnalysis.lib.distances import minimize_vectors

class PBNN_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running PBNN simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, diabats, couplings, plumed_call=False, **kwargs):
        """
        Parameters
        -----------
        diabats : list
            List of diabat objects
        couplings : list
            List of couplings objects
        plumed_call : bool
            whether to use Plumed or not
        """

        Calculator.__init__(self, **kwargs)
        self.diabats = diabats
        self.couplings = couplings

        self.universe = Universe(self.diabats[0].openmm.pdbtemplate)

        self.plumed_call = plumed_call

        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom

        self.jobtype = kwargs.get("jobtype")
        #If running the test job, then can just grab OpenMM components
        if self.jobtype == "MD":
            self.omm_terms = []
        else:
            self.omm_terms = ['CustomBondForce', 'NonbondedForce', 'CustomNonbondedForce', 'DrudeForce'] 
        self.frame = 0

    def diagonalize(self, diabat_energies, coupling_energies):
        """
        Forms matrix and diagonalizes using np to obtain ground-state
        eigenvalue and eigenvector.

        Parameters
        -----------
        diabat_energies : list
            List containing the energies of the diabatic states
        coupling_energies : list
            List containing the coupling energies between diabatic states

        Returns
        -----------
        eig[l_eig] : np.ndarray
            Ground-state eigenvalue
        eigv[:, l_eig] : np.ndarray
            Ground-state eigenvector
        """
        num_states = len(diabat_energies)
        hamiltonian = np.zeros((num_states, num_states))
        for i, energy in enumerate(diabat_energies):
            hamiltonian[i, i] = energy
        for i, energy in enumerate(coupling_energies):
            index = self.couplings[i].couplings_loc
            hamiltonian[index[0], index[1]] = energy
            hamiltonian[index[1], index[0]] = energy

        eig, eigv = np.linalg.eig(hamiltonian)
        l_eig = np.argmin(eig)
        eigv = np.around(eigv, decimals=10)
        return eig[l_eig], eigv[:, l_eig]

    def calculate_forces(self, diabat_forces, coupling_forces, ci):
        """
        Uses Hellmann-Feynman theorem to calculate forces on each atom.

        Parameters
        -----------
        diabat_forces : list
            List containing the forces for each diabat
        coupling_forces : list
            List containing the forces from the coupling elements
        ci : np.ndarray
            ground-state eigenvector

        Returns
        -----------
        np.ndarray
            Forces calculated from Hellman-Feynman theorem
        """
        num_states = len(diabat_forces)
        hamiltonian_force = np.zeros((num_states, num_states), dtype=np.ndarray)
        for i, force in enumerate(diabat_forces):
            hamiltonian_force[i, i] = force

        for i, force in enumerate(coupling_forces):
            index = self.couplings[i].couplings_loc
            hamiltonian_force[index[0], index[1]] = force
            hamiltonian_force[index[1], index[0]] = force
        
        total_forces = 0
        for i in range(num_states):
            for j in range(num_states):
                total_forces += ci[i] * ci[j] * hamiltonian_force[i, j]

        return total_forces

    def get_field(self, atoms):
        """
        Get field on reacting complex atoms from external solvent

        Parameters
        -----------
        atoms : object
            ASE atoms
        """
      
        self.universe.coord.positions = atoms.get_positions().astype(np.float32)
        #Get the field from the solvent
        if len(self.universe.residues) != len(self.diabats[0].openmm.exclude_intra_res):
            #Get all residues in the reactive complex
            sel = 'resid '
            for res in range(len(self.diabats[0].openmm.exclude_intra_res)-1):
                sel += f'{self.diabats[0].openmm.exclude_intra_res[res]+1} '
            sel += f'{self.diabats[0].openmm.exclude_intra_res[-1]+1}'

            #Select reactive complex atoms
            react_complex = self.universe.select_atoms(sel)

            #Select solvent atoms
            solvent = self.universe.select_atoms('not '+sel)

            #Center of geometry of the reactive complex
            react_centroid = react_complex.center_of_geometry(wrap=True)
            #Center of geometry of solvent molecules
            solvent_centroids = solvent.center_of_geometry(compound='residues', wrap=True)

            #Distances from reactive complex centroid to solvent centroids
            dist_arr = distances.distance_array(react_centroid, solvent_centroids, box=self.universe.dimensions)[0]
            #OpenMM electrostatics cutoff
            cutoff = self.diabats[0].openmm.cutoff * 10

            #Indices of residues within cutoff in dist_arr
            residues = np.where(dist_arr < cutoff)[0]
            
            #Get the actual residue indices from residues
            solvent_residues = [s.resid for s in solvent.residues]
            sel = 'resid '
            for res in range(len(residues)-1): sel += f'{solvent_residues[residues[res]]} '
            sel += f'{solvent_residues[residues[-1]]}'
            
            #Select all atoms in the embedding residues
            embedding_list = self.universe.select_atoms(sel)
            
            #Distance from reactive complex atoms to embedding atoms
            dist_arr = distances.distance_array(react_complex.positions, embedding_list.positions, box=self.universe.dimensions)

            #Form array from reactive complex positions shape (num reactive complex atoms, num embedding atoms, 3)
            react_complex_positions = np.repeat(react_complex.positions, len(embedding_list), axis=0).reshape(react_complex.positions.shape[0], len(embedding_list), 3)
            
            #Minimum image displacements
            disp = react_complex_positions[:] - embedding_list.positions
            disp = disp.reshape(disp.shape[0]*disp.shape[1], 3)
            disp = minimize_vectors(disp, box=self.universe.dimensions)
            disp = disp.reshape(len(react_complex), len(embedding_list), 3)

            #Get charges from OpenMM for reactive complex and embedding atoms
            charges = np.asarray(self.diabats[0].openmm.get_charges())
            react_complex_indices = [atom.index for atom in react_complex]
            embedding_indices = [atom.index for atom in embedding_list]
            react_complex_charges = charges[react_complex_indices]
            embedding_charges = charges[embedding_indices]

            #Get the electrostatic potential on all atoms from the diabat 1 perspective
            dist_arr /= units.Bohr
            disp /= units.Bohr
            atom_potential = embedding_charges * 1/dist_arr[:] * 2625.5
            atom_potential_d1 = atom_potential.sum(axis=1)
            
            #Get the energy on all atoms from diabat 1
            #q_prod = atom_potential_d1 * react_complex_charges
            #energy_d1 = q_prod.sum(axis=0)

            #Reshape embedding charges array so that it can be multiplied with displacement array to form field
            #embedding_charges = np.repeat(embedding_charges[None], disp.shape[0], axis=0)[:,:,None]
            #field = disp[:] * embedding_charges * 2625.5 * 1/dist_arr[:,:,None]**3 / units.Bohr 
            #field_d1 = -field.sum(axis=1)

            #Get the forces from electrostatics
            #forces_d1 = field_d1 * react_complex_charges[:,None]

            #Get charges for diabat 2
            #charges = np.asarray(self.diabats[1].openmm.get_charges())

            #Reorder atom_potential_d1 to atom_potential_d2
            atom_potential_d2 = atom_potential_d1[self.diabats[1].reorder_graph.diabat_reorder.astype(int)]

            #react_complex_charges = charges[react_complex_indices]

            #Get diabat 2 electrostatic potential energy
            #q_prod = atom_potential_d2 * react_complex_charges
            #energy_d2 = q_prod.sum(axis=0)

            #Get field and forces for diabat 2
            #field_d2 = field_d1[self.diabats[1].reorder_graph.diabat_reorder.astype(int)]
            #forces_d2 = field_d2 * react_complex_charges[:,None]
            self.atom_potential = [atom_potential_d1, atom_potential_d2]
            self.total_potential = atom_potential_d1
            
    def set_field(self, potential=None, field=None, atom_potential=None, total_potential=None):
        """
        Set various properties from external field if they are already computed

        Parameters
        -----------
        potential : optional or np.ndarray
            energy from external potential
        field : optional or np.ndarray
            forces from external field
        atom_potential : optional or list
            electrial potential on each atom from the solvent for both diabats
        total_potential : optional or np.ndarray
            also the electrical potential on each atom (just in the canonical H11 order)
        """
        self.potential = potential
        self.field = field
        self.atom_potential = atom_potential
        self.total_potential = total_potential

    @staticmethod
    def wrap(position_array, residue_atom_lists, box):
        """
        Make sure residues are wrapped within the first minimum image

        Parameters
        -----------
        position_array : np.ndarray
            positions
        residue_atom_lists : list
            List of lists containing atoms in each residue
        box : object
            Cell vectors
        """
        #box = atoms.get_cell()
        #print(box)
        inv_box = np.linalg.inv(box)
        #print(inv_box)
        new_positions = np.zeros_like(position_array)
        for residue in residue_atom_lists:
            residue_positions = position_array[residue]
            residue_centroid = [
                sum([position[k] for position in residue_positions])
                / len(residue) for k in range(3)
            ]
            inv_centroid = residue_centroid @ inv_box
            mask = np.floor(inv_centroid)
            diff = -mask @ box
            for atom in residue:
                new_positions[atom] = position_array[atom] + diff
        return new_positions

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Parameters
        -----------
        atoms : object
            ASE atoms object
        properties : list
            List of properties that are being called
        system_changes : list
            List of changes that have taken place within the atoms object since called last time
        """

        result = {}
        #If plumed call, then the wrap step should already be taken care of
        if any(atoms.pbc) and not self.plumed_call:
            positions = self.wrap(atoms.get_positions(), self.diabats[0].openmm.real_atom_res_list, atoms.get_cell())
            atoms.positions = positions
        
        Calculator.calculate(self, atoms, properties, system_changes)
       
        print("PBNN", self.frame)

        self.get_field(atoms)
        diabat_energies = []
        diabat_forces = []
        for i, diabat in enumerate(self.diabats):
            energy, forces = diabat.compute_energy_force(atoms, potential=self.potential[i], field=self.field[i], atom_potential=self.atom_potential[i])
            diabat_energies.append(energy)
            diabat_forces.append(forces)
        
        coupling_energies = []
        coupling_forces = []
        for coupling in self.couplings:
            coupling_force = np.zeros_like(diabat_forces[-1])
            energy, forces = coupling.compute_energy_force(atoms, coupling_force, self.total_potential)
            coupling_energies.append(energy)
            coupling_forces.append(forces)
        
        energy, ci = self.diagonalize(diabat_energies, coupling_energies)
        forces = self.calculate_forces(diabat_forces, coupling_forces, ci)

        self.frame += 1
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class PBNN_Interface:
    """
    Set up PBNN calculator and simulation
    """
    def __init__(self, atoms, diabats, coupling, tmp, plumed_command, **kwargs):
        """
        Parameters
        -----------
        atoms : object
            ASE atoms object
        diabats : list
            List of diabat classes
        coupling : list
            List of coupling classes
        tmp : str
            Where to store output files
        plumed_command : list
            List containing Plumed command
        """

        if isinstance(atoms, list):
            self.atoms = atoms[-1]
        else:
            self.atoms = read(atoms)
        
        self.diabats = diabats

        res_list = self.diabats[0].openmm.res_list()

        if isinstance(coupling, list):
            self.couplings = coupling
        else:
            self.couplings = [coupling]

        self.potential = [0, 0]
        field_h11 = np.zeros_like(self.atoms.get_positions())
        field_h22 = np.zeros_like(self.atoms.get_positions())
        self.field = np.append(field_h11, field_h22, axis=0)

        self.tmp = tmp 
        if not os.path.isdir(self.tmp): os.makedirs(self.tmp)

        self.plumed_command = plumed_command

        #Need to wrap PBNN calculator into Plumed calculator if using Plumed
        if self.plumed_command:
            calc = PBNN_Hamiltonian(self.diabats, self.couplings, plumed_call=True, **kwargs)
            self.calculator = Plumed(calc, self.plumed_command, 1.0, atoms=self.atoms, kT=300.0*units.kB, log=f'{self.tmp}/colvar.dat', res_list=res_list)
        else:
            self.calculator = PBNN_Hamiltonian(self.diabats, self.couplings, **kwargs)

        self.atoms.set_calculator(self.calculator)
        self.rewrite = kwargs.get('rewrite_log', True)
        self.jobtype = kwargs.get('jobtype')

        self.md = False

    def set_mol_positions(self, positions):
        """
        Parameters 
        ------------
        positions : np.ndarray
            positions to set for the molecule object
        """
        self.atoms.positions = positions

    def calculate_single_point(self, potential=None, field=None, atom_potential=None, total_potential=None):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. 

        Parameters
        -----------
        potential : None or np.ndarray
            electrical potential energy on each atom
        field : None or np.ndarray
            forces from the external potential
        atom_potential : None or list
            potential on each atom in each diabat
        total_potential : None or np.ndarray
            potential on each atom
        """

        if potential:
            if self.plumed_command:
                self.atoms.calc.calc.set_field(potential, field, atom_potential, total_potential)
            else:
                self.atoms.calc.set_field(potential, field, atom_potential, total_potential)

        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        self.atoms.energy = energy
        self.atoms.forces = forces
        return energy, forces

    def setup_test(self, data, energy, forces, e_potential, e_field, name=None):
        """
        Assemble data for testing PBNN on a set of data

        Parameters
        -----------
        data : list
            List of atoms objects
        energy : np.ndarray
            Contains energy for each configuration
        forces : np.ndarray
            Contains forces for each configuration
        e_potential : np.ndarray
            Electrical potential for each configuration
        e_field : np.ndarray
            Electrical field for each configuration
        name : None or str
            Filename for saving the output
        """
        
        self.data = data
        self.energy = energy
        self.forces = forces
        self.e_potential = e_potential
        self.e_field = e_field
        self.name = name

    def create_system(self, 
            name, 
            time_step=1.0, 
            temp=300, 
            temp_init=None, 
            restart=False, 
            store=1, 
            ensemble='nvt', 
            friction=0.001,
            remove_translation=True,
            remove_rotation=True):
        """
        Parameters
        -----------
        name : str
            Name for output files.
        time_step : float, optional
            Time step in fs for simulation.
        temp : float, optional
            Temperature in K for NVT simulation.
        temp_init : float, optional
            Optional different temperature for initialization than thermostate set at.
        restart : bool, optional
            Determines whether simulation is restarted or not,
            determines whether new velocities are initialized.
        store : int, optional
            Frequency at which output is written to log files.
        nvt : bool, optional
            Determines whether to run NVT simulation, default is False.
        friction : float, optional
            friction coefficient in fs^-1 for Langevin integrator
        remove_translation : bool
            Whether to remove velocities due to translation of simulation box
        remove_rotation : bool
            Whether to remove velocities due to rotation of simulation box
        """
        if temp_init is None: temp_init = temp
        if not self.md or restart:
            MaxwellBoltzmannDistribution(self.atoms, temp_init * units.kB)

            if remove_translation:
                Stationary(self.atoms)
            if remove_rotation:
                ZeroRotation(self.atoms)

        ensemble = ensemble.lower()

        if ensemble == 'nve':
            self.md = VelocityVerlet(self.atoms, time_step * units.fs)
        elif ensemble == 'nvt':
            self.md = Langevin(self.atoms, time_step * units.fs, temperature_K=temp, friction=friction/units.fs)

        logfile = os.path.join(self.tmp, "{}.log".format(name))
        self.trajfile = os.path.join(self.tmp, "{}.traj".format(name))
        if self.rewrite and os.path.isfile(logfile) and os.path.isfile(self.trajfile):
            os.remove(logfile)
            os.remove(self.trajfile)

        logger = MDLogger(self.md, self.atoms, logfile, stress=False, peratom=False, header=True, mode="w")
        trajectory = Trajectory(self.trajfile, "w", self.atoms)
        self.md.attach(logger, interval=store)
        self.md.attach(trajectory.write, interval=store)

        potential_h11 = 0
        potential_h22 = 0
        res_list = self.diabats[0].openmm.res_list()
        monomer_A_h11 = res_list[0]
        monomer_B_h11 = res_list[1]

        potential_A_h11 = np.zeros((self.atoms.get_positions()[monomer_A_h11].shape[0]))
        potential_B_h11 = np.zeros((self.atoms.get_positions()[monomer_B_h11].shape[0]))

        res_list = self.diabats[1].openmm.res_list()
        monomer_A_h22 = res_list[0]
        monomer_B_h22 = res_list[1]

        potential_A_h22 = np.zeros((self.atoms.get_positions()[monomer_A_h22].shape[0]))
        potential_B_h22 = np.zeros((self.atoms.get_positions()[monomer_B_h22].shape[0]))
        potential_solv = np.zeros((self.atoms.get_positions().shape[0]))

        potential = [potential_h11, potential_h22]
        field = np.zeros((self.atoms.get_positions().shape[0], 3))

        atom_potential = [np.append(potential_A_h11, potential_B_h11, axis=0), np.append(potential_A_h22, potential_B_h22, axis=0)]

        if self.jobtype == 'MD':
            potential = [None, None]
            field = [None, None]
            if self.plumed_command:
                self.atoms.calc.calc.set_field(potential=potential, atom_potential=atom_potential, field=field, total_potential=potential_solv)
            else:
                self.atoms.calc.set_field(potential=potential, atom_potential=atom_potential, field=field, total_potential=potential_solv)
        else:
            self.atoms.calc.set_field(potential, field, atom_potential, potential_solv)

    def run_test(self):
        """
        Assemble forces from PBNN and save reference forces to a npy file
        """
        test_energy = []
        test_forces = []
        
        for i, atoms in enumerate(self.data):
            print(i)
            self.set_mol_positions(atoms.get_positions())

            field_h11 = np.zeros_like(atoms.get_positions())
            field_h22 = np.zeros_like(atoms.get_positions())

            potential_h11 = 0
            potential_h22 = 0

            res_list = self.diabats[0].openmm.res_list()
            monomer_A_h11 = res_list[0]
            monomer_B_h11 = res_list[1]

            potential_A_h11 = np.zeros((atoms.get_positions()[monomer_A_h11].shape[0]))
            potential_B_h11 = np.zeros((atoms.get_positions()[monomer_B_h11].shape[0]))

            res_list = self.diabats[1].openmm.res_list()
            monomer_A_h22 = res_list[0]
            monomer_B_h22 = res_list[1]

            potential_A_h22 = np.zeros((atoms.get_positions()[monomer_A_h22].shape[0]))
            potential_B_h22 = np.zeros((atoms.get_positions()[monomer_B_h22].shape[0]))
            potential_solv = np.zeros((atoms.get_positions().shape[0]))

            if len(self.e_potential[i]):
                potential = self.e_potential[i]
                potential = np.asarray(potential)
                potential = potential[None]
                charges = self.diabats[0].openmm.get_charges()
                for atom_potential in potential:
                    potential_A_h11 += atom_potential[monomer_A_h11]
                    potential_B_h11 += atom_potential[monomer_B_h11]
                    potential_solv += atom_potential
                    #Should rewrite this so there's no loop...
                    for p, charge in zip(atom_potential, charges):
                        potential_h11 += p*charge

                charges = self.diabats[1].openmm.get_charges()
                for atom_potential in potential:
                    new_atoms, reorder_list = self.diabats[1].reorder_graph.reorder(atoms)
                    reorder_potential = atom_potential[reorder_list]
                    potential_A_h22 += reorder_potential[monomer_A_h22]
                    potential_B_h22 += reorder_potential[monomer_B_h22]
                    for p, charge in zip(reorder_potential, charges):
                        potential_h22 += p*charge

            if len(self.e_field[i]):
                charges = self.diabats[0].openmm.get_charges()
                e_field = self.e_field[i]
                e_field = np.asarray(e_field)
                e_field = e_field[None]
                for atom_field in e_field:
                    for j, (f, charge) in enumerate(zip(atom_field, charges)):
                        field_h11[j] -= f*charge

                charges = self.diabats[1].openmm.get_charges()
                for atom_field in e_field:
                    new_atoms, reorder_list = self.diabats[1].reorder_graph.reorder(atoms)
                    reorder_field = atom_field[reorder_list]
                    for j, (f, charge) in enumerate(zip(reorder_field, charges)):
                        field_h22[j] -= f*charge
           
            field = [field_h11, field_h22]
            atom_potential_h11 = np.append(potential_A_h11, potential_B_h11, axis=0)
            atom_potential_h22 = np.append(potential_A_h22, potential_B_h22, axis=0)
            atom_potential = [atom_potential_h11, atom_potential_h22]
            potential = [potential_h11, potential_h22]
            energy, forces = self.calculate_single_point(potential=potential, 
                    field=field, 
                    atom_potential=atom_potential, 
                    total_potential=potential_solv)
            #Convert to kj/mol
            energy *= 96.486
            forces *= 96.486
            test_energy.append(energy)
            test_forces.append(forces)

        test_energy = np.asarray(test_energy)
        self.energy = np.asarray(self.energy)
        test_forces = np.asarray(test_forces)
        self.forces = np.asarray(self.forces)

        test_energy = test_energy.reshape(test_energy.shape[0], 1)
        self.energy = self.energy.reshape(self.energy.shape[0], 1)
        test_forces = test_forces.reshape(test_forces.shape[0], test_forces.shape[1]*3)
        self.forces = self.forces.reshape(self.forces.shape[0], self.forces.shape[1]*3)
        
        if self.name:
            fname = f'{self.name}'
        else:
            fname = 'test'

        np.save(f"ref_{fname}_energy.npy", self.energy)
        np.save(f"test_{fname}_energy.npy", test_energy)
        np.save(f"ref_{fname}_forces.npy", self.forces)
        np.save(f"test_{fname}_forces.npy", test_forces)

    def run_md(self, steps):
        """
        Run MD simulation.

        Parameters
        -----------
        steps : int
            Number of MD steps
        """
        for i in range(steps):
            self.md.run(1)


