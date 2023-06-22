#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM/MM system to propogate dynamic simulations.

Manages the interactions between the QM and MM subsytems through their
respective interface objects.
"""
import os
import sys

import numpy as np
import ase
import ase.io
import ase.md
import ase.md.velocitydistribution as ase_md_veldist
import ase.optimize
from ase import units
from ase.calculators.singlepoint import SinglePointDFTCalculator

from .qmmm_hamiltonian import *
from .logger import Logger
from .plumed_calculator import Plumed

class QMMMEnvironment:
    """
    Sets up and runs the QM/MM/MD simulation.  
    
    Serves as an interface to OpenMM, Psi4 and ASE.  
    
    Based off of the
    ASE_MD class from SchNetPack.

    Parameters
    ----------
    atoms: str
        Location of input structure to create an ASE Atoms object.
    tmp: str
        Location for tmp directory.
    openmm_interface: OpenMMInterface object
        A pre-existing OpenMM interface with which the QM/MM system may
        communicate.
    psi4_interface: Psi4Interface object
        A pre-existing Psi4 interface with which the QM/MM system may
        communicate.
    qm_atoms_list: list of int
        List of atom indices representing the QM subsystem under
        investigation.
    embedding_cutoff: float
        Cutoff for analytic charge embedding, in Angstroms.
    rewrite_log: bool, Optional, default=True
        Determine whether or not an existing log file gets overwritten.
    plumed_command : list
        list of commands for Plumed
    dataset_builder : obj
        during the simulation, build a dataset for PB/NN neural networks.
    """

    def __init__(self, atoms, tmp, openmm_interface, psi4_interface, 
                 qm_atoms_list, embedding_cutoff, rewrite_log=True, plumed_command=None, dataset_builder=None):
        # Set working directory.

        self.plumed_command = plumed_command
        if plumed_command:
            self.plumed_call = True
        else:
            self.plumed_call = False

        self.tmp = tmp
        
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)
        
        # Define Atoms object.
        if isinstance(atoms, ase.Atoms):
            self.atoms = atoms
        elif isinstance(atoms, list):
            self.atoms = atoms[-1]
        else:
            self.atoms = ase.io.read(atoms)

        # We freeze atoms with mass=0.  The best way to do this is to exclude these atoms from the ase atoms object
        if self.atoms.get_global_number_of_atoms() != openmm_interface.atoms_nofreeze :
            # don't use openmm_interface.get_masses because that will return masses in ase datastructure format
            # rather we want all masses (including frozen atoms) which are in openmm_interface._masses_all
            mask_freeze_atoms = self.ase_atoms_remove_frozen( self.atoms , openmm_interface._masses_all , openmm_interface._mass_threshold , qm_atoms_list )
            openmm_interface.mask_freeze_atoms = mask_freeze_atoms 
        else:
            openmm_interface.mask_freeze_atoms = np.array([True])

        # Collect OpenMM interface and respective atom properties.
        self.openmm_interface = openmm_interface

        # create residue_atom_lists using mask_freeze_atoms
        self.openmm_interface.set_residue_atom_lists( openmm_interface.mask_freeze_atoms )
        self._residue_atom_lists = self.openmm_interface.get_residue_atom_lists()
    
        # Collect Psi4 interface and set qm_atoms_list
        self.psi4_interface = psi4_interface
        self.psi4_interface.set_qm_atoms_list(qm_atoms_list)
        self.psi4_interface.set_chemical_symbols(np.asarray(self.atoms.get_chemical_symbols()))

        print("bypassing minimization call ...")
        #self.psi4_interface.generate_ground_state_energy(self.atoms.get_positions())
        self.qm_atoms_list = qm_atoms_list
        self.embedding_cutoff = embedding_cutoff
        self.rewrite = rewrite_log
        self.dataset_builder = dataset_builder

    def create_system(self, name, time_step=1.0, temp=300, temp_init=None,
                      restart=False, write_freq=1, ensemble="nve",
                      friction=0.001, remove_translation=False, 
                      remove_rotation=False, embed_electrode=False):
        """
        Creates the simulation environment for ASE.

        Parameters
        ----------
        name: str
            Name for output files.
        time_step: float, Optional, default=1.0
            Time step in fs for simulation.
        temp: float, Optional, default=300
            Temperature in K for NVT simulation.
        temp_init: float, Optional, default=None
            Optional different temperature for initialization than
            thermostate set at.
        restart: bool, Optional, default=False
            Determines whether simulation is restarted or not, 
            determines whether new velocities are initialized.
        write_freq: int, Optional, default=1
            Frequency at which output is written to log files.  Taken to
            be every x number of time steps.
        ensemble: str, Optional, default="nve"
            Determines which integrator to use given an ensemble of
            variables.
        friction: float, Optional, default=0.001
            Friction coefficient in fs^-1 for Langevin integrator in the
            NVT ensemble.
        remove_translation: bool, Optional, default=False
            Determine whether to zero center of mass translation.
        remove_rotation: bool, Optional, default=False
            Determine whether to zero center of mass rotation.
        """
        # Set initial simulation options.
        if temp_init is None:
            temp_init = temp
        if restart:
            ase_md_veldist.MaxwellBoltzmannDistribution(
                self.atoms,
                temp_init * ase.units.kB,
                rng=np.random.default_rng(seed=42),
            )
        if remove_translation:
            ase_md_veldist.Stationary(self.atoms)
        if remove_rotation:
            ase_md_veldist.ZeroRotation(self.atoms)
        if ensemble.lower() == "nve":
            self.md = ase.md.VelocityVerlet(self.atoms, time_step * ase.units.fs)
        elif ensemble.lower() == "nvt":
            self.md = ase.md.Langevin(self.atoms,
                                      time_step * ase.units.fs,
                                      temperature_K=temp,
                                      friction=friction / ase.units.fs)
        elif ensemble.lower() == "npt":
            print("NPT ensemble is not currently implemented...")
            sys.exit()
        else:
            print("""Unrecognized ensemble input to QMMMEnvironment
                  initialization.""")
            sys.exit()

        #  temperature, timestep, friction are not import in OpenMM, since OpenMM integrator is not used ...

        # Supercede OpenMMInterface settings.
        #self.openmm_interface.set_temperature(temp)
        # ASE takes friction in fs^1, whereas OpenMM takes friction in ps^-1.
        #self.openmm_interface.set_friction(friction * 1000)
        # ASE takes time step in fs, whereas OpenMM takes time step in ps.
        #self.openmm_interface.set_time_step(time_step / 1000)

        self.openmm_interface.create_subsystem()
        # These are currently implemented only for real atoms, not all
        # particles (such as virtual sites or drudes).

        self.atoms.set_masses(self.openmm_interface.get_masses())
        self.atoms.charges = self.openmm_interface.get_charges()

        self.openmm_interface.set_positions(self.atoms.get_positions())
        # setup electrode embedding if requested
        if embed_electrode:
            self.openmm_interface.initialize_electrode_embedding()

        self.psi4_interface.set_charges(self.openmm_interface.get_charges())
        mm_atoms_list = self.openmm_interface.get_mm_atoms_list()

        if self.plumed_call:
            res_list = self.openmm_interface.get_residue_atom_lists()
            # Define Calculator.
            calculator = QMMMHamiltonian(self.openmm_interface, self.psi4_interface,
                                     self.qm_atoms_list, mm_atoms_list, self.embedding_cutoff,
                                     self._residue_atom_lists, embed_electrode, plumed_call=self.plumed_call, database_builder=self.dataset_builder)
            self.calculator = Plumed(calculator, self.plumed_command, 1.0, atoms=self.atoms, kT=300.0*units.kB, log=f'{self.tmp}/colvar.dat', res_list=res_list)
        else:
            self.calculator = QMMMHamiltonian(self.openmm_interface, self.psi4_interface,
                                     self.qm_atoms_list, mm_atoms_list, self.embedding_cutoff,
                                     self._residue_atom_lists, embed_electrode, plumed_call=self.plumed_call, database_builder=self.dataset_builder)
        
        self.atoms.set_calculator(self.calculator)
        
        self.name = name       
        # Determine ouput files for simulation.
        log_file = os.path.join(self.tmp, "{}.log".format(name))
        traj_file = os.path.join(self.tmp, "{}.dcd".format(name))
        if (self.rewrite
                and os.path.isfile(log_file)
                and os.path.isfile(traj_file)):
            os.remove(log_file)
            os.remove(traj_file)
       
        logger = Logger(log_file)
        logger.write("="*30 + "QM/MM/MD Log" + "="*30 + "\n")

        if self.plumed_call:
            self.logger = logger
            self.atoms.calc.calc.logger = logger
        else:
            self.logger = logger
            self.atoms.calc.logger = logger
        
        self.openmm_interface.logger = logger
        self.openmm_interface.generate_reporter(traj_file, append=restart)

    def write_atoms(self, name, ftype="xyz", append=False):
        """
        Write out current system structure.

        Parameters
        ----------
        name: str
            Name of the output file.
        ftype: str, Optional, defalt="xyz"
            Determines output file format.
        append: bool, Optional, default=False
            Determine whether to append to existing output file or not.
        """
        path = os.path.join(self.tmp, "{}.{}".format(name, ftype))
        ase.io.write(path, self.atoms, format=ftype, append=append)

    def calculate_single_point(self):
        """
        Perform a single point energy and force computation.

        Returns
        -------
        energy: Numpy array
            The energy calculated for the system by ASE
        forces: Numpy array
            The forces calculated for the system by ASE
        """
        self.openmm_interface.set_positions(self.atoms.get_positions())
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        return energy, forces

    def run_test(self, atoms):
        self.openmm_interface.create_subsystem()

        self.atoms.set_masses(self.openmm_interface.get_masses())
        self.atoms.charges = self.openmm_interface.get_charges()

        self.openmm_interface.set_positions(self.atoms.get_positions())

        self.psi4_interface.set_charges(self.openmm_interface.get_charges())
        mm_atoms_list = self.openmm_interface.get_mm_atoms_list()

        calculator = QMMMHamiltonian(self.openmm_interface, self.psi4_interface,
                                    self.qm_atoms_list, mm_atoms_list, self.embedding_cutoff,
                                    self._residue_atom_lists, False, database_builder=self.dataset_builder)

        self.atoms.set_calculator(calculator)

        #traj_file = os.path.join(self.tmp, "{}.dcd".format(name))
        traj = []
        for frame in atoms:
            self.atoms = frame
            self.atoms.set_masses(self.openmm_interface.get_masses())
            self.atoms.charges = self.openmm_interface.get_charges()

            self.atoms.set_calculator(calculator)
            try:
                energy, forces = self.calculate_single_point()
                energy *= 96.487
                forces *= 96.487
    
                calc = SinglePointDFTCalculator(frame, energy=energy, forces=forces)
                frame.calc = calc
                traj.append(frame)
            except:
                print("Calculation failed")

        ase.io.write(f"{self.tmp}.traj", traj)

    def run_md(self, steps):
        """
        Run MD simulation.

        Parameters
        ----------
        steps : int
            Number of MD steps.
        """
        for step in range(steps):
            self.md.run(1)
            self.write_atoms(self.name, ftype='traj')

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer.

        Parameters
        ----------
        fmax: float, Optional, default=1.0e-2
            Maximum residual force change.
        steps: int
            Maximum number of steps.
        """
        name = "optimization"
        optimize_file = os.path.join(self.tmp, name)
        optimizer = ase.optimize.QuasiNewton(self.atoms,
                                             trajectory="%s.traj" % optimize_file,
                                             restart="%s.pkl" % optimize_file,)
        optimizer.run(fmax, steps)
        self.write_atoms(name)

    def ase_atoms_remove_frozen(self, atoms , masses, mass_threshold, qm_atoms_list):
        """
        recreates the self.atoms ase object
        to remove atoms with mass=0 to be frozen
        Parameters
        ----------
        atoms: ase.atoms
            The ase.atoms object
        masses: Numpy array
            The masses from openmm
        mass_threshold: float
            threshold for freezing atoms
        qm_atoms_list: list
            list of atoms in the qm region
        Returns
        -------
        mask_freeze_atoms: Numpy array
            mask indicating which atoms are frozen as "False" 
        """

        # first make sure all the qm atoms are indexed before the first frozen atom
        # we don't want the qm atom indices to shift which would cause confusion...
        max_qm_index = max(qm_atoms_list)
        for i in range(max_qm_index+1):
            if masses[i] < mass_threshold:
                print(" all qm atoms should be indexed before any frozen atoms !! ")
                sys.exit()

        # generate mask for later use 
        mask_freeze_atoms=[]
        for i in range(len(masses)):
            if masses[i] < mass_threshold:
                mask_freeze_atoms.append(False)
            else:
                mask_freeze_atoms.append(True)
       
        # pop atoms off the atoms object with zero mass
        # loop backward through atoms object so that pop doesn't
        # change lower indices ...
        for i in range(len(masses)):
            j = len(masses)-i-1
            if masses[j] < mass_threshold:
                atoms.pop(j)
       
        return np.array(mask_freeze_atoms)
