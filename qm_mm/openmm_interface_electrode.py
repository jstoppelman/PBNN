#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMM interface to model the MM subsystem of the QM/MM system.
"""
import numpy as np
import openmm.app
import openmm
import simtk.unit
import sys
sys.path.append("../")
from qm_mm_md_electrode_embed.openmm_interface import OpenMMInterface
from electrode_exclusions import *
from electrode_classes import *

class OpenMMInterface_electrode(OpenMMInterface):
    """
    OpenMM interface for the MM subsytem. 

    Parameters
    ----------
    pdb_file: string
        The directory and filename for the PDB file.
    residue_xml_list: list of str
        The directories and filenames for the XML topology file.
    ff_xml_list: list of str
        The directories and filenames for the XML force field file.
    platform: str
        One of four available platforms on which to run OpenMM.  These 
        include "Reference", "CPU", "CUDA", and "OpenCL".
    temperature: float, Optional, default=300
        The temperature of the system in Kelvin.
    temperature_drude: float, Optional, default=1
        The temperature at which to equilibrate the Drude particles in
        Kelvin.
    friction: float, Optional, default=1
        Determines coupling to the heat bath in Langevin Integrator in
        inverse picoseconds.
    friction_drude: float, Optional, default=1
        Determines coupling to the heat bath during equilibration of the
        Drude particles in inverse picoseconds.
    time_step: float, Optional, default=0.001
        Time step at which to perform the simulation.
    charge_threshold: float, Optional, default=1e-6
        Threshold for charge convergeance.
    nonbonded_cutoff: floar, Optional, default=1.4
        Cutoff for nonbonded interactions, such as Lennard-Jones and 
        Coulomb, in nanometers.
    """

    def __init__(self, pdb_file, residue_xml_list, ff_xml_list, platform, qm_atoms_list, **kwargs): 

        # fix this, remove any optional input in super ...
        super().__init__(pdb_file, residue_xml_list, ff_xml_list, platform, qm_atoms_list, **kwargs)

        if 'total_charge_per_electrode' in kwargs :
            total_charge_per_electrode = float(kwargs['total_charge_per_electrode'])
        else:
            print( 'need to input total_charge_per_electrode' )
            sys.exit()

        if 'QM_umbrella_settings' in kwargs :
            self.QM_umbrella = True
            self.QM_umbrella_settings = kwargs['QM_umbrella_settings']
        else :
            self.QM_umbrella = False

        # initialize electrodes  !! FIX hardcoded electrode names ...
        self.Cathode = electrode(self.pdb, self.nonbonded_force, electrode_name = 'CAT')
        self.Anode = electrode(self.pdb, self.nonbonded_force, electrode_name = 'ANO')

        self.Cathode.determine_surface_atoms()
        self.Anode.determine_surface_atoms()

        # Set charges of the interior surface atoms of each electrode (positive for cathode, negative for anode)
        self.Cathode.set_surface_charge(total_charge_per_electrode)
        self.Anode.set_surface_charge(-1*total_charge_per_electrode)

        # this generates exclusions for intra-electrode interactions
        cathode_list=[]
        for atom in self.Cathode.electrode_atoms:
            cathode_list.append( atom.atom_index )
        anode_list=[]
        for atom in self.Anode.electrode_atoms:
            anode_list.append( atom.atom_index )
        
        exclusion_Electrode_NonbondedForce(cathode_list, cathode_list, self.system_mm, self.custom_nonbonded_force, self.nonbonded_force)
        exclusion_Electrode_NonbondedForce(anode_list, anode_list, self.system_mm, self.custom_nonbonded_force, self.nonbonded_force)

        # setup umbrella potential on QM system if requested ...
        if self.QM_umbrella:
            resname = self.QM_umbrella_settings[0]
            atomtype = self.QM_umbrella_settings[1]
            kz = self.QM_umbrella_settings[2]
            z0 = self.QM_umbrella_settings[3]
            ZForce = CustomExternalForce("0.5*kz*periodicdistance(x,y,z,x,y,z0)^2")
            ZForce.setName('ZForce')
            ZForce.addGlobalParameter('kz', kz)
            ZForce.addPerParticleParameter('z0')

            umbrella_atom_index = None
            for res in self.modeller.topology.residues():
                if res.name == resname:
                    for atom in res._atoms:
                        if atom.name == atomtype:
                            umbrella_atom_index = atom.index
            if umbrella_atom_index == None:
                print ('atom index was not found for QM umbrella potential')
                sys.exit()

            ZForce.addParticle(umbrella_atom_index, [z0])
            self.system_lj.addForce(ZForce)
          

        # this is run in super().create_subsystem() , which is called in super().__init__(), but
        # we need to reset self._charges_all here since we've changed charges on electrodes ...
        charges = []
        for i in range(self.system_mm.getNumParticles()):
            (q, sig, eps) = self.nonbonded_force.getParticleParameters(i)
            charges.append(q._value)
        self._charges_all = charges

       
    # this setups datastructures for embedding electrode charges in qm/mm
    def initialize_electrode_embedding(self):
        # setup electrode charge/position datastructures for surface atoms.
        # note we can do positions only once since they are frozen ...
        self.electrode_embedding_positions_Ang=[]
        self.electrode_embedding_charges=[]
        
        # store these in Angstrom, so convert from stored nanometers
        # first cathode
        for atom in self.Cathode.surface_atoms:    
            self.electrode_embedding_charges.append( atom.charge._value )
            self.electrode_embedding_positions_Ang.append( np.array( [ self._positions[atom.atom_index][0]._value*self.nm_to_angstrom , 
                                                                   self._positions[atom.atom_index][1]._value*self.nm_to_angstrom ,
                                                                   self._positions[atom.atom_index][2]._value*self.nm_to_angstrom ] ) )
        # now anode
        for atom in self.Anode.surface_atoms:    
            self.electrode_embedding_charges.append( atom.charge._value )
            self.electrode_embedding_positions_Ang.append( np.array( [ self._positions[atom.atom_index][0]._value*self.nm_to_angstrom , 
                                                                   self._positions[atom.atom_index][1]._value*self.nm_to_angstrom ,
                                                                   self._positions[atom.atom_index][2]._value*self.nm_to_angstrom ] ) )


class atom_MM(object):
    def __init__(self, element, charge, atom_index, x, y, z):
        self.element = element
        self.charge  = charge
        self.atom_index = atom_index
        self.x = x; self.y=y; self.z=z

class electrode(object):
    def __init__(self, pdb, nbondedForce, electrode_name):
        self.electrode_name = electrode_name
        self.topology = pdb.getTopology()
        self.positions = pdb.getPositions()
        self.nbondedForce = nbondedForce
        self.electrode_atoms = []

        for res in self.topology.residues():
            if res.name == self.electrode_name:
                for atom in res._atoms:
                    element = atom.element
                    (q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.index)
                    atom_object = atom_MM(element.symbol, q_i._value, atom.index, self.positions[atom.index][0]._value, self.positions[atom.index][1]._value, self.positions[atom.index][2]._value)
                    self.electrode_atoms.append(atom_object)

    def get_total_charge(self):
        sumQ = 0.0
        for atom in self.electrode_atoms:
            sumQ += atom.charge
        return sumQ

    def determine_surface_atoms(self):
        dims = [f._value for f in self.topology.getUnitCellDimensions()]
        atoms_z_coords = [atom.z for atom in self.electrode_atoms]
        self.surface_atoms = []
        if self.electrode_name == 'CAT':
            # set atoms with max z to be surface atoms
            max_z = 0
            for atom in self.electrode_atoms:
                if atom.z > max_z:
                    max_z = atom.z
            for atom in self.electrode_atoms:
                if atom.z == max_z:
                    self.surface_atoms.append(atom)
        elif self.electrode_name == 'ANO':
            # set atoms with min z to be surface atoms
            min_z = dims[2]
            for atom in self.electrode_atoms:
                if atom.z < min_z:
                    min_z = atom.z
            for atom in self.electrode_atoms:
                if atom.z == min_z:
                    self.surface_atoms.append(atom)
        #print (self.electrode_name,[(atom.element, atom.atom_index) for atom in self.surface_atoms])

    def set_surface_charge(self, total_charge):
        print (total_charge, len(self.surface_atoms), elementary_charge)
        #print (self.electrode_name,[(atom.element, atom.atom_index) for atom in self.surface_atoms])
        charge_per_atom = (total_charge / len(self.surface_atoms))*elementary_charge
        for atom in self.surface_atoms:
            (q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.atom_index)
            self.nbondedForce.setParticleParameters(atom.atom_index, charge_per_atom, sig, eps)
            atom.charge = charge_per_atom
            #print ('elec_name:', self.electrode_name, 'elem:', atom.element, 'index:', atom.atom_index, 'charge:', atom.charge, 'nbf_charge:', q_i)


        return

