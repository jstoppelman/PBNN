#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Psi4 interface to model the QM subsystem of the QM/MM system.
"""
import sys
import numpy as np
import psi4
import psi4.core


class Psi4Interface:
    """
    Psi4 interface for the QM subsystem.

    Parameters
    ----------
    basis_set: str
        Name of desired Psi4 basis set.
    functional: str
        Name of desired Psi4 density functional.
    quadrature_spherical: int
        Number of spherical (angular and azimuthal) points for the 
        exchange-correlation functional quadrature.
    quadrature_radial: int
        Number of radial points for the exchange-correlation functional
        quadrature.
    qm_charge: int
        Charge of the QM system in proton charge
    qm_spin: int
        Spin of the QM system.
    n_threads: int, Optional, default=1
        Number of threads across which to parallelize the QM calculation.
    read_guess: bool, Optional, default=True
        Determine whether to base calculations on previous wavefunction
        objects (default) or on the Psi4 default SAD.
    """

    def __init__(self, basis_set, functional, quadrature_spherical,
                 quadrature_radial, qm_charge, qm_spin, n_threads=1,
                 read_guess=False, pruning=False, SOSCF=False):
        self.basis = basis_set
        self.dft_functional = functional
        self.dft_spherical_points = quadrature_spherical
        self.dft_radial_points = quadrature_radial
        self.qm_charge = qm_charge
        self.qm_spin = qm_spin
        self.n_threads = n_threads
        self.read_guess = read_guess
        self.wfn = False
        # Their is an opportunity to make qm_atoms_list a property.
        self.set_qm_atoms_list(None)
        #self.set_ground_state_energy(None)
        self.set_ground_state_energy(0.0)
        psi4.set_num_threads(self.n_threads)
        # Set scf type to be density functional.
        self.scf_type = 'df'
        if pruning:
            self.opts = {'basis': self.basis,
                         'dft_spherical_points': self.dft_spherical_points,
                         'dft_radial_points': self.dft_radial_points,
                         'scf_type': self.scf_type,
                         'dft_pruning_scheme': pruning }

        else:
            self.opts = {'basis': self.basis,
                         'dft_spherical_points': self.dft_spherical_points,
                         'dft_radial_points': self.dft_radial_points,
                         'scf_type': self.scf_type}
        if SOSCF:
            psi4.core.set_local_option('SCF', 'SOSCF', 'True')
    
    def generate_geometry(self, embedding_list, offsets, positions, embed_electrode = False , 
                          embedding_list_electrode_pos = [] , embedding_list_electrode_charge = [] ,
                          delr_vector_list_electrode = [] ):
        """
        Create the geometry string to feed into the Psi4 calculation.

        Parameters
        ----------
        embedding_list: list of list of int
            Integer indices of atoms within the analytic embedding cutoff
            grouped by residue.
        offsets: list of list of float
            Offset to add onto the position of each atom grouped
            by residue.
        positions: Numpy array
            Array of atom positions from the ASE Atoms object.
        """
        # Check spin state of the QM molecule.
        if self.qm_spin > 1:
           psi4.core.set_local_option('SCF', 'REFERENCE', 'UKS')
        # Set the field of electrostatically embedded charges.
        chargefield = []
        for residue, offset in zip(embedding_list, offsets):
            for atom in residue:
                position = [(positions[atom,k] + offset[k])*1.88973 for k in range(3)]
                chargefield.append([self._charges[atom],[position[0],position[1],position[2]]])
        # if we need to embed electrode charges ...
        if embed_electrode :
            for charge , position , offset in zip( embedding_list_electrode_charge , embedding_list_electrode_pos , delr_vector_list_electrode ):
                position = [(position[k] + offset[k])*1.88973 for k in range(3)]
                chargefield.append([charge ,[position[0],position[1],position[2]]])
                
        self.chargefield = chargefield
        #psi4.core.set_global_option('PRINT', 5)
        # Construct geometry string.
        geometrystring = """\n"""
        qm_centroid = [sum([positions[i][k] for i in self._qm_atoms_list]) 
                       / len(self._qm_atoms_list) for k in range(3)]
        for atom in self._qm_atoms_list:
            position = [positions[atom,k] - qm_centroid[k] for k in range(3)]
            geometrystring = (geometrystring + " "
                              + str(self._chemical_symbols[atom]) + " " 
                              + str(position[0]) + " " 
                              + str(position[1]) + " " 
                              + str(position[2]) + "\n")
        geometrystring = (geometrystring + str(self.qm_charge) + " " 
                          + str(self.qm_spin) + "\n")
        # Do not reorient molecule.
        geometrystring = geometrystring + "noreorient\nnocom\n"
        self.geometry = psi4.geometry(geometrystring)

    def generate_ground_state_energy(self, positions):
        """
        Calculate the ground state energy of the QM subsystem.

        Parameters
        ----------
        position_array : Numpy array
            Array of atom positions from the ASE Atoms object.
        """
        # Check spin state of the QM molecule.
        if self.qm_spin > 1:
           psi4.core.set_local_option('SCF', 'REFERENCE', 'UKS')
        psi4.core.set_global_option('PRINT', 0)
        # Construct geometry string.
        geometrystring = """\n"""
        for atom in self._qm_atoms_list:
            position = positions[atom]
            geometrystring = (geometrystring
                              + str(self._chemical_symbols[atom]) + " " 
                              + str(position[0]) + " " 
                              + str(position[1]) + " " 
                              + str(position[2]) + "\n")
        geometrystring = (geometrystring + str(self.qm_charge) + " " 
                          + str(self.qm_spin) + "\n")
        # Do not reorient molecule.
        geometrystring = geometrystring + "noreorient\nnocom\n"
        geometry = psi4.geometry(geometrystring)
        psi4.set_options(self.opts)
        psi4_energy = psi4.optimize(self.dft_functional, molecule=geometry)
        self.set_ground_state_energy(psi4_energy)

    def compute_energy(self):
        """
        Calculates the energy and forces for the QM atoms.

        These forces are the intramolecular forces acting on amongst the
        QM atoms and the the electrostatic forces acting on the QM atoms
        from the extended environment.

        Returns
        -------
        psi4_energy: Numpy array
            QM subsystem energy.
        psi4_forces: Numpy array
            Forces acting on QM atoms.
        """
        psi4.set_options(self.opts)
        # Check for wavefunction if read_guess is True.
        if self.wfn and self.read_guess:
            self.wfn.to_file(self.wfn.get_scratch_filename(180))
            psi4.core.set_local_option('SCF', 'GUESS', 'READ')
        if len(self.chargefield) == 0:
            print(self.opts)
            sys.exit()
            (psi4_energy, psi4_wfn) = psi4.energy(
                self.dft_functional,
                return_wfn=True,
                molecule=self.geometry,
                options=self.opts
            )
            psi4_forces = psi4.gradient(
                self.dft_functional,
                ref_wfn=psi4_wfn,
                options=self.opts
            )
        
        else:
            (psi4_energy, psi4_wfn) = psi4.energy(
                self.dft_functional,
                return_wfn=True,
                external_potentials=self.chargefield,
                molecule=self.geometry,
                options=self.opts
            )
            psi4_forces = psi4.gradient(
                self.dft_functional,
                external_potentials=self.chargefield,
                ref_wfn=psi4_wfn,
                options=self.opts
            )
        self.wfn = psi4_wfn
        # Convert energy to kJ/mol and forces to kJ/mol/Angstrom
        psi4_energy = (psi4_energy - self._ground_state_energy)*2625.5
        psi4_forces = -np.asarray(psi4_forces)*2625.5*1.88973
        return psi4_energy, psi4_forces

    def get_qm_atoms_list(self):
        """
        Get the list of QM atom indices.

        Returns
        -------
        qm_atoms_list: list of int
            Indices of the QM atoms within the ASE Atoms object.
        """
        return self._qm_atoms_list

    def set_qm_atoms_list(self, qm_atoms_list):
        """
        Set the list of QM atom indices.

        Parameters
        ----------
        qm_atoms_list: list of int
            Indices of the QM atoms within the ASE Atoms object.
        """
        self._qm_atoms_list = qm_atoms_list

    def set_charges(self, charges):
        """
        Sets the charges for atoms in the ASE Atoms object.

        Parameters
        ----------
        charges: Numpy array
            Partial charges in proton charge by atom index.
        """
        self._charges = charges

    def get_chemical_symbols(self):
        """
        Gets the chemical symbols for atoms in the ASE Atoms object.

        Returns
        -------
        chemical_symbols: Numpy array
            Chemical symbols by atom index.
        """
        return self._chemical_symbols

    def set_chemical_symbols(self, chemical_symbols):
        """
        Gets the charges from atoms in the system

        Parameters
        ----------
        chemical_symbols: Numpy array
            Chemical symbols by atom index.
        """
        self._chemical_symbols = chemical_symbols

    def get_ground_state_energy(self):
        """
        Get the absolute ground-state energy for the QM subsystem in au.

        Returns
        -------
        ground_state_energy: Numpy array
            The absolute ground state energy of the QM subsystem.
        """
        return self._ground_state_energy

    def set_ground_state_energy(self, ground_state_energy):
        """
        Set the absolute ground-state energy for the QM subsystem in au.

        Parameters
        ----------
        ground_state_energy: Numpy array
            The absolute ground state energy of the QM subsystem.
        """
        self._ground_state_energy = ground_state_energy
