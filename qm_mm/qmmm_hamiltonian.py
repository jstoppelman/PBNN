#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASE Calculator to combine QM and MM forces and energies.
"""
import sys

import numpy as np
import ase
from ase.calculators.calculator import Calculator, all_changes

from .utils import *

class QMMMHamiltonian(Calculator):
    """ 
    ASE Calculator.

    Modeled after SchNetPack calculator.

    Parameters
    ----------
    openmm_interface: OpenMMInterface object
        OpenMMInterface object containing all the needed info for
        getting forces from OpenMM.
    psi4_interface: Psi4Interface object
        Psi4Interface object containing all the needed info for getting
        forces from Psi4.
    qm_atoms_list: list of int
        List containing the integer indices of the QM atoms
    embedding_cutoff: float
        Cutoff distance, in Angstroms, within which molecules will be
        electrostatically embedded in the QM calculation.
    residue_atom_lists: list of list of int
        Residue list containing lists of atom indices in each residue
    **kwargs: dict
        Additional args for ASE base calculator
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, openmm_interface, psi4_interface, qm_atoms_list, mm_atoms_list,
                 embedding_cutoff, residue_atom_lists, embed_electrode, plumed_call=False, database_builder=None, **kwargs):
        
        Calculator.__init__(self, **kwargs)
        self.openmm_interface = openmm_interface
        self.psi4_interface = psi4_interface
        self.qm_atoms_list = qm_atoms_list
        self.mm_atoms_list = mm_atoms_list
        self.embedding_cutoff = embedding_cutoff
        self.residue_atom_lists = residue_atom_lists
        self.embed_electrode = embed_electrode
        if embed_electrode:
            self.electrode_embedding_charges = openmm_interface.electrode_embedding_charges
            self.electrode_embedding_positions_Ang = openmm_interface.electrode_embedding_positions_Ang

        #self.has_periodic_box = self.openmm_interface.has_periodic_box
        self.has_periodic_box = True
        self.energy_units = ase.units.kJ / ase.units.mol
        self.forces_units = ase.units.kJ / ase.units.mol / ase.units.Angstrom
        self.frame = 0
        self.plumed_call = plumed_call
        self.database_builder = database_builder

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """
        Obtains the total energy and forces using the above interfaces.

        Parameters
        ------------
        atoms: ASE Atoms object, Optional, default=None
            Atoms object containing coordinates.
        properties: list of str, Optional, default=['energy']
            Not used.
        system_changes: list, Optional, 
                default=ase.calculators.calculator.all_changes
            List of changes for ASE.
        """

        result = {}
       
        if self.has_periodic_box and not self.plumed_call:
            new_positions = self.wrap(atoms.get_positions(), self.residue_atom_lists, atoms.get_cell())
            atoms.positions = new_positions
        
        # Set gemoetry for the QM/MM Psi4 calculation.
        self.embed_electrostatics(atoms.positions, atoms.get_cell())
        if self.embed_electrode:
            self.embed_electrostatics_electrode(atoms.positions, atoms.get_cell())
            self.psi4_interface.generate_geometry(self.embedding_list, 
                                              self.delr_vector_list,
                                              atoms.positions,
                                              embed_electrode = True,
                                              embedding_list_electrode_pos = self.embedding_list_electrode_pos ,
                                              embedding_list_electrode_charge = self.embedding_list_electrode_charge ,
                                              delr_vector_list_electrode = self.delr_vector_list_electrode )
        else: 
            self.psi4_interface.generate_geometry(self.embedding_list, 
                                              self.delr_vector_list,
                                              atoms.positions)
        
        Calculator.calculate(self, atoms)

        self.openmm_interface.set_positions(atoms.get_positions())
        self.openmm_interface.embedding_list = self.embedding_list
        self.openmm_interface.delr_vector_list = self.delr_vector_list
       
        openmm_energy, openmm_forces = self.openmm_interface.compute_energy()
        psi4_energy, psi4_forces = self.psi4_interface.compute_energy()

        qm_forces = psi4_forces[self.qm_atoms_list,:]
        qm_index = np.arange(0, len(self.qm_atoms_list), 1).astype(int)
        em_forces = np.delete(psi4_forces, qm_index, axis=0)

        # Add Psi4 electrostatic forces and energy onto OpenMM forces
        # and energy for QM atoms.

        total_forces = np.zeros_like(openmm_forces)
        
        total_forces += openmm_forces
        total_forces[self.qm_atoms_list, :] += qm_forces

        # Remove double-counting from embedding forces and energy.
        j = 0
        qm_centroid = [sum([atoms.positions[i][j] for i in self.qm_atoms_list])
                       / len(self.qm_atoms_list) for j in range(3)]
        dc_energy = 0.0
        # if freezing atoms, need new data structure for forces on embedding atoms since they won't line up with openmm_forces ..
        if not self.openmm_interface.mask_freeze_atoms.all():
            temp_forces = np.zeros_like(atoms.positions)

        e_field_total = []
        e_potential_total = []
        e_potential = []
        e_field = []
        embed = []
        for residue, offset in zip(self.embedding_list, self.delr_vector_list):
            for atom in residue:
                co_forces = [0,0,0]
                e_field = []
                e_potential = []
                distance = []
                for i in self.qm_atoms_list:
                    x = (atoms.positions[atom][0] + offset[0] - (atoms.positions[i][0] - qm_centroid[0])) * 1.88973
                    y = (atoms.positions[atom][1] + offset[1] - (atoms.positions[i][1] - qm_centroid[1])) * 1.88973
                    z = (atoms.positions[atom][2] + offset[2] - (atoms.positions[i][2] - qm_centroid[2])) * 1.88973
                    dr = (x**2 + y**2 + z**2)**0.5

                    q_prod = atoms.charges[i] * atoms.charges[atom]
                    co_forces[0] += 1.88973 * 2625.5 * x * q_prod * dr**-3
                    co_forces[1] += 1.88973 * 2625.5 * y * q_prod * dr**-3
                    co_forces[2] += 1.88973 * 2625.5 * z * q_prod * dr**-3
                    dc_energy += 2625.5 * q_prod * dr**-1

                    field_x = 1.88973 * 2625.5 * x * atoms.charges[atom] * dr**-3
                    field_y = 1.88973 * 2625.5 * y * atoms.charges[atom] * dr**-3
                    field_z = 1.88973 * 2625.5 * z * atoms.charges[atom] * dr**-3
                    
                    #print(atoms.charges[i]*field_x, atoms.charges[i]*field_y, atoms.charges[i]*field_z)
                    e_field.append([field_x, field_y, field_z])
                    e_potential.append(2625.5 * atoms.charges[atom] * dr**-1)
                
                e_field_total.append(e_field)
                e_potential_total.append(e_potential)
                if not self.openmm_interface.mask_freeze_atoms.all():
                    temp_forces[atom] += em_forces[j][i]
                    temp_forces[atom] -= co_forces
                else:
                    total_forces[atom] += em_forces[j]
                    total_forces[atom] -= co_forces
                j += 1

        e_potential_total = np.asarray(e_potential_total)
        e_potential_total = e_potential_total.sum(axis=0)
        e_field_total = np.asarray(e_field_total)
        e_field_total = e_field_total.sum(axis=0)
        
        # now remove double counting energy from electrode embedding.  Forces can be ignored because electrode atoms are frozen...
        if self.embed_electrode:
            for charge , position , offset in zip( self.embedding_list_electrode_charge , self.embedding_list_electrode_pos , self.delr_vector_list_electrode ):
                for i in self.qm_atoms_list:
                    x = (position[0] + offset[0] - (atoms.positions[i][0] - qm_centroid[0])) * 1.88973
                    y = (position[1] + offset[1] - (atoms.positions[i][1] - qm_centroid[1])) * 1.88973
                    z = (position[2] + offset[2] - (atoms.positions[i][2] - qm_centroid[2])) * 1.88973
                    dr = (x**2 + y**2 + z**2)**0.5
                    q_prod = atoms.charges[i] * charge
                    dc_energy += 2625.5 * q_prod * dr**-1
        
        total_energy = openmm_energy + psi4_energy - dc_energy
        if hasattr(self, 'logger'):
            self.log_energy(atoms, psi4_energy, dc_energy, total_energy)

        self.frame += 1
        result["energy"] = total_energy * self.energy_units
        # need to mask forces if freezing atoms for md integration
        if not self.openmm_interface.mask_freeze_atoms.all():
            result["forces"] = (total_forces[self.openmm_interface.mask_freeze_atoms] + temp_forces) * self.forces_units
        else:
            result["forces"] = total_forces * self.forces_units

        if self.database_builder:
            self.database_builder.write_db(atoms, self.frame-1, psi4_energy, qm_forces, e_field_total, e_potential_total)

        self.results = result

    @staticmethod
    def wrap(position_array, residue_atom_lists, box):
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

    def embed_electrostatics(self, positions, box):
        """
        Collects the indices of atoms which fall within the embedding
        cutoff of the centroid of the QM atoms.

        Parameters
        ----------
        positions: NumPy array
            Array of atom positions within the periodic box
        box: list of list of float
            Cell object from ASE, which contains the box vectors.
        """
        qm_centroid = [sum([positions[i][j] for i in self.qm_atoms_list])
                       / len(self.qm_atoms_list) for j in range(3)]
        embedding_list = []
        qm_drude_list = []
        delr_vector_list = []
        embedding_res_list = []
        for i_res, residue in enumerate(self.residue_atom_lists):
            # Get the least mirror distance between the QM molecule
            # centroid and the centroid of the current molecule.
            nth_centroid = [sum([positions[i][j] for i in residue]) 
                            / len(residue) for j in range(3)]
            # Legacy embedding.
            #nth_centroid = [positions[residue[0]][j] for j in range(3)]
            r_vector = least_mirror_distance(qm_centroid, 
                                             nth_centroid,
                                             box)
            distance = sum([r_vector[i]**2 for i in range(3)])**(0.5)
            if distance < self.embedding_cutoff:
                if not any([atom in self.qm_atoms_list for atom in residue]):
                    embedding_list.append(residue)
                    for r in residue: embedding_res_list.append(r)
                    delr_vector_list.append([r_vector[k]
                                             - nth_centroid[k] for k in range(3)])
                # If atoms are not in the qm_atoms_list and they share 
                # the same residue as the QM atoms, then they must be 
                # drudes from the QM atoms.
                else:
                    qm_drude_list = np.setdiff1d(np.array(residue), 
                                                 np.array(self.qm_atoms_list))
        self.embedding_list = embedding_list
        self.delr_vector_list = delr_vector_list
        self.qm_drude_list = qm_drude_list
        embedding_res_list = np.asarray(embedding_res_list)
    
    def embed_electrostatics_electrode(self, positions, box):
        """
        """
        qm_centroid = [sum([positions[i][j] for i in self.qm_atoms_list])
                       / len(self.qm_atoms_list) for j in range(3)]
        embedding_list_electrode_charge = []
        embedding_list_electrode_pos = []
        delr_vector_list_electrode = []
        # loop over charged electrode atoms
        for charge, position in zip(self.electrode_embedding_charges , self.electrode_embedding_positions_Ang):
            # Get the least mirror distance between the QM molecule
            # centroid and the electrode atom 
            r_vector = least_mirror_distance(qm_centroid, 
                                             position,
                                             box)
            distance = sum([r_vector[i]**2 for i in range(3)])**(0.5)
            if distance < self.embedding_cutoff:
                embedding_list_electrode_pos.append( position )
                embedding_list_electrode_charge.append( charge )
                delr_vector_list_electrode.append([r_vector[k]
                                             - position[k] for k in range(3)])
        self.embedding_list_electrode_pos = embedding_list_electrode_pos
        self.embedding_list_electrode_charge = embedding_list_electrode_charge
        self.delr_vector_list_electrode = delr_vector_list_electrode

    def log_energy(self, atoms, psi4_energy, dc_energy, total_energy):
        self.logger.write("\n" + "-"*29 + "Frame " + "0"*(8-len(str(self.frame))) + str(self.frame) + "-"*29 + "\n")
        category = "Kinetic Energy"
        value = str(atoms.get_kinetic_energy()*96.4869)
        left, right = value.split(".")
        self.logger.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")

        category = "Psi4 Energy"
        value = str(psi4_energy)
        left, right = value.split(".")
        self.logger.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
        category = "Correction Energy"
        value = str(-dc_energy)
        left, right = value.split(".")
        self.logger.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")

        category = "Total Energy"
        value = str(atoms.get_kinetic_energy()*96.4869 + total_energy)
        left, right = value.split(".")
        self.logger.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")

