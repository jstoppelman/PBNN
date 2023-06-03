from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import Calculator, all_changes

#**********************************************
# this routine uses OpenMM to evaluate energies 
# Can use with Drude oscillators (DrudeSCF procedure)
#**********************************************

class OpenMM_ForceField(Calculator):
    """
    Setup OpenMM simulation object with template pdb
    and xml force field files
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, pdbtemplate, residuexml, saptxml, platformtype='CPU', Drude_hyper_force=True, exclude_intra_res=[], cutoff=None, **kwargs):
        """
        pdbtemplate : str
            Path to PDB file that contains the residues for a specific diabat
        residuexml : str
            Path to xml file with residue definitions
        saptxml : str
            Path to xml file with force field parameters
        platformtype : str
            OpenMM platform types
        Drude_hyper_force : bool
            Apply an anharmonic restraining potential if using Drude oscillators
        exclude_intra_res : list
            Indices of residues for which all intramolecular interactions should be excluded
        cutoff : float or None
            Whether a cutoff is used for electrostatics and vdW. Cutoff should be in nm
        """
        Calculator.__init__(self, **kwargs)
        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        Topology().loadBondDefinitions(residuexml)
        self.pdbtemplate = pdbtemplate
        self.pdb = PDBFile(self.pdbtemplate)  # this is used for topology, coordinates are not used...
        self.Drude_hyper_force = Drude_hyper_force
        if Drude_hyper_force:
            self.integrator = DrudeSCFIntegrator(0.001*picoseconds) # Use the SCF integrator to optimize Drude positions    
            self.integrator.setMinimizationErrorTolerance(0.01)
        else:
            self.integrator = VerletIntegrator(0.001*picoseconds)
        self.real_atom_topology, positions = self.pdb.topology, self.pdb.positions
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = ForceField(saptxml)
        self.modeller.addExtraParticles(self.forcefield)  # add Drude oscillators
        self.platformtype = platformtype

        self.has_periodic_box = self.pdb.topology.getPeriodicBoxVectors()
        if cutoff:
            self.cutoff = float(cutoff)
            self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff*nanometer, constraints=None, rigidWater=False)
        else:
            self.cutoff = None
            # by default, no cutoff is used, so all interactions are computed.  This is what we want for gas phase PES...no Ewald!!
            self.system = self.forcefield.createSystem(self.modeller.topology, constraints=None, rigidWater=True)

        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]

        self.charges = []
        for i in range(nbondedForce.getNumParticles()):
            self.charges.append(nbondedForce.getParticleParameters(i)[0]/elementary_charge)
        
        self.exclude_intra_res = exclude_intra_res
        self.diabat_resids = deepcopy(exclude_intra_res)

        self.create_simulation()

        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        self.positions = state.getPositions()
        #Determines if OpenMM should only get the energy for specific components/custom force classes
        self.energy_expression = kwargs.get("energy_expression", None)
        self.components = kwargs.get("components", None)

    def create_simulation(self):
        """
        Setup simulation object
        """

        # Obtain a list of real atoms excluding Drude particles for obtaining forces later
        # Also set particle mass to 0 in order to optimize Drude positions without affecting atom positions
        self.realAtoms = []
        self.masses = []
        self.allMasses = []
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i)/dalton > 1.0:
                self.realAtoms.append(i)
                self.masses.append(self.system.getParticleMass(i)/dalton)
            elif i in real_element:
                self.realAtoms.append(i)

            self.allMasses.append(self.system.getParticleMass(i)/dalton)
            self.system.setParticleMass(i,0)

        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        customBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        #Determine whether to use PBCs for customBond and nbondedForce
        if self.has_periodic_box:
            customBondForce.setUsesPeriodicBoundaryConditions(True)
            nbondedForce.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        else:
            nbondedForce.setNonbondedMethod(NonbondedForce.NoCutoff)
        customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
        customNonbondedForce.setUseLongRangeCorrection(True)

        # add "hard wall" hyper force to Drude/parent atoms to prevent divergence with SCF integrator...
        if self.Drude_hyper_force:
            self.add_Drude_hyper_force()

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)

        #Platform used for OpenMM
        self.platform = Platform.getPlatformByName(self.platformtype)
        if self.platformtype == 'CPU':
            self.properties = {}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'Reference':
            self.properties = {}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'CUDA':
            #self.properties = {'CudaPrecision': 'double'}
            self.properties = {'CudaPrecision': 'mixed'}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        elif self.platformtype == 'OpenCL':
            #self.properties = {'OpenCLPrecision': 'double'}
            self.properties = {'OpenCLPrecision': 'mixed'}
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)

        #Residue list for only real atoms
        real_atom_res_list = []
        residue_topology_types = {}
        for res in self.real_atom_topology.residues():
            c_res = []
            if res.name not in residue_topology_types.keys():
                residue_topology_types[res.name] = res
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            real_atom_res_list.append(c_res)
        self.real_atom_res_list = real_atom_res_list
        self.residue_topology_types = residue_topology_types

        #Residue list of all the atoms in the simulation
        all_atom_res_list = []
        res_names = []
        for res in self.simulation.topology.residues():
            c_res = []
            for i in range(len(res._atoms)):
                c_res.append(res._atoms[i].index)
            all_atom_res_list.append(c_res)
            res_names.append(res.name)
        self.res_names = res_names
        self.all_atom_res_list = all_atom_res_list

        #Exclude residues present in exclude_intra_res
        if self.exclude_intra_res is not None:
            self.exclude_atom_list = [z for i in self.exclude_intra_res for z in all_atom_res_list[i]]
            self.diabat_exclude = [z for i in self.diabat_resids for z in self.all_atom_res_list[i]]
            self.add_exclusions_monomer_intra()
            self.nnRealAtoms = [z for i in self.exclude_intra_res for z in self.real_atom_res_list[i]]
            self.nn_res_list = []
            for i in self.exclude_intra_res:
                self.nn_res_list.append(self.real_atom_res_list[i])

        if self.Drude_hyper_force:
            #Collection of Drude pairs
            self.get_drude_pairs()

    def add_exclusions_monomer_intra(self):
        """
        Setup intramolecular exclusions for the reacting atoms
        """

        self.harmonicBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicBondForce][0]
        self.harmonicAngleForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicAngleForce][0]
        if [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce]:
            self.periodicTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce][0]
        if [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == RBTorsionForce]:
            self.rbTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == RBTorsionForce][0]
        self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        self.customBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]
        if self.Drude_hyper_force:
            self.drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        
        #Zero energies from intramolecular forces for residues with neural networks
        for i in range(self.harmonicBondForce.getNumBonds()):
            p1, p2, r0, k = self.harmonicBondForce.getBondParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list:
                k = Quantity(0, unit=k.unit)
                self.harmonicBondForce.setBondParameters(i, p1, p2, r0, k)
        
        for i in range(self.harmonicAngleForce.getNumAngles()):
            p1, p2, p3, r0, k = self.harmonicAngleForce.getAngleParameters(i)
            if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list:
                k = Quantity(0, unit=k.unit)
                self.harmonicAngleForce.setAngleParameters(i, p1, p2, p3, r0, k)

        if hasattr(self, 'periodicTorsionForce'):
            for i in range(self.periodicTorsionForce.getNumTorsions()):
                p1, p2, p3, p4, period, r0, k = self.periodicTorsionForce.getTorsionParameters(i)
                if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list or p4 in self.exclude_atom_list:
                    k = Quantity(0, unit=k.unit)
                    self.periodicTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, period, r0, k)

        if hasattr(self, 'rbTorsionForce'):
            for i in range(rbTorsionForce.getNumTorsions()):
                p1, p2, p3, p4, c1, c2, c3, c4, c5, c6 = self.rbTorsionForce.getTorsionParameters(i)
                if p1 in self.exclude_atom_list or p2 in self.exclude_atom_list or p3 in self.exclude_atom_list or p4 in self.exclude_atom_list:
                    c1 = Quantity(0, unit=c1.unit)
                    c2 = Quantity(0, unit=c2.unit)
                    c3 = Quantity(0, unit=c3.unit)
                    c4 = Quantity(0, unit=c4.unit)
                    c5 = Quantity(0, unit=c5.unit)
                    c6 = Quantity(0, unit=c6.unit)
                    self.rbTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, c1, c2, c3, c4, c5, c6)

        if self.Drude_hyper_force:
            # map from global particle index to drudeforce object index
            particleMap = {}
            for i in range(self.drudeForce.getNumParticles()):
                particleMap[self.drudeForce.getParticleParameters(i)[0]] = i

        # can't add duplicate ScreenedPairs, so store what we already have
        flagexceptions = {}
        for i in range(self.nbondedForce.getNumExceptions()):
            (particle1, particle2, charge, sigma, epsilon) = self.nbondedForce.getExceptionParameters(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexceptions[string1]=1
            flagexceptions[string2]=1

        if hasattr(self, 'customNonbondedForce'):
            # can't add duplicate customNonbonded exclusions, so store what we already have
            flagexclusions = {}
            for i in range(self.customNonbondedForce.getNumExclusions()):
                (particle1, particle2) = self.customNonbondedForce.getExclusionParticles(i)
                string1=str(particle1)+"_"+str(particle2)
                string2=str(particle2)+"_"+str(particle1)
                flagexclusions[string1]=1
                flagexclusions[string2]=1

        print(' adding exclusions ...')

        # add all intra-molecular exclusions, and when a drude pair is
        # excluded add a corresponding screened thole interaction in its place
        for res in self.simulation.topology.residues():
            for i in range(len(res._atoms)-1):
                for j in range(i+1,len(res._atoms)):
                    (indi,indj) = (res._atoms[i].index, res._atoms[j].index)
                    # here it doesn't matter if we already have this, since we pass the "True" flag
                    self.nbondedForce.addException(indi,indj,0,1,0,True)
                    # make sure we don't already exclude this customnonbond
                    string1=str(indi)+"_"+str(indj)
                    string2=str(indj)+"_"+str(indi)
                    if hasattr(self, 'customNonbondedForce'):
                        if string1 in flagexclusions or string2 in flagexclusions:
                            continue
                        else:
                            self.customNonbondedForce.addExclusion(indi,indj)
                    if self.Drude_hyper_force:
                        # add thole if we're excluding two drudes
                        if indi in particleMap and indj in particleMap:
                            # make sure we don't already have this screened pair
                            if string1 in flagexceptions or string2 in flagexceptions:
                                continue
                            else:
                                drudei = particleMap[indi]
                                drudej = particleMap[indj]
                                self.drudeForce.addScreenedPair(drudei, drudej, 2.0)
        
        # now reinitialize to make sure changes are stored in context
        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simulation.context.reinitialize()
        self.simulation.context.setPositions(positions)

    def add_Drude_hyper_force(self):
        """
        this method adds a "hard wall" hyper bond force to
        parent/drude atoms to prevent divergence using the
        Drude SCFIntegrator ...
        """

        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]

        hyper = CustomBondForce('step(r-rhyper)*((r-rhyper)*khyper)^powh')
        hyper.addGlobalParameter('khyper', 100.0)
        hyper.addGlobalParameter('rhyper', 0.02)
        hyper.addGlobalParameter('powh', 6)
        self.system.addForce(hyper)
        for i in range(drudeForce.getNumParticles()):
            param = drudeForce.getParticleParameters(i)
            drude = param[0]
            parent = param[1]
            hyper.addBond(drude, parent)

    def res_list(self):
        """
        Return dictionary of indices for each monomer in the dimer. 
        """
        res_dict = []
        k = 0
        for res in self.pdb.topology.residues():
            res_list = []
            for i in range(len(res._atoms)):
                res_list.append(res._atoms[i].index)
            res_dict.append(res_list)
        return res_dict

    def setReactAtom(self, react_atom, react_residue):
        """
        Give OpenMM the reacting atom index in the diabat and 
        the residue index of the reacting residue

        Parameters
        -----------
        react_atom : int
            Atom index of the reacting atom
        react_residue : int
            Residue index of the reacting residue
        """
        self.react_atom = react_atom
        self.react_residue = react_residue

    def get_masses(self):
        """
        Return masses
        """
        return self.masses

    def get_charges(self):
        """
        Return charges
        """
        return self.charges

    def get_drude_pairs(self):
        """
        Get Drude atom parent pairs
        """
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.drudePairs = []
        for i in range(drudeForce.getNumParticles()):
            parms = drudeForce.getParticleParameters(i)
            self.drudePairs.append((parms[0], parms[1]))

    def set_initial_positions(self, xyz):
        """
        Set the initial OpenMM positions
        Only important for when using Drude oscillators

        Parameters
        -----------
        xyz : np.ndarray
            array containing xyz positions
        """
        self.xyz_pos = xyz
        for i in range(len(self.xyz_pos)):
            # update pdb positions
            self.positions[i] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        # now update positions in modeller object
        self.modeller = Modeller(self.pdb.topology, self.positions)
        # add dummy site and shell initial positions
        self.modeller.addExtraParticles(self.forcefield)
        self.simulation.context.setPositions(self.modeller.positions)
        self.positions = self.modeller.positions

    def set_xyz(self, xyz):
        """
        Set current OpenMM positions for this frame

        Parameters
        -----------
        xyz : np.ndarray
            positions array
        """

        self.xyz_pos = xyz
        self.initial_positions = self.simulation.context.getState(getPositions=True).getPositions()
        for i in range(len(self.realAtoms)):
            self.positions[self.realAtoms[i]] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        if self.Drude_hyper_force:
            self.drudeDisplacement()
        self.simulation.context.setPositions(self.positions)

    def drudeDisplacement(self):
        """
        Get displacement of Drude parent atoms from the previous step
        """
        for i in range(len(self.drudePairs)):
            disp = self.positions[self.drudePairs[i][1]] - self.initial_positions[self.drudePairs[i][1]]
            self.positions[self.drudePairs[i][0]] += disp

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Get OpenMM energy and forces (either total energy or force components if force components were given in kwargs

        Parameters
        -----------
        atoms : ASE Atoms object
            current atoms
        properties : list
            List of properties that are being called
        system_changes : list
            List of changes that have taken place within the atoms object since called last time
        """ 
        self.results = {}
        
        self.set_xyz(atoms.get_positions())
        
        if self.components:
            energy, forces = self.compute_energy_component(self.components, energy_expression=self.energy_expression)
        else:
            energy, forces = self.compute_energy()
        
        self.results["energy"] = energy
        self.results["forces"] = forces

    def compute_energy(self):
        """
        Compute the energy for a particular configuration

        Returns
        ----------
        eOPENMM : np.ndarray
            OpenMM energy in kj/mol
        OPENMM_forces : np.ndarray
            OpenMM forces in kj/mol/A
        """
        # integrate one step to optimize Drude positions.  Note that atoms won't move if masses are set to zero
        self.simulation.step(1)
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()
        
        # get energy
        state = self.simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        eOPENMM = state.getPotentialEnergy()/kilojoule_per_mole
        OPENMM_forces = state.getForces(asNumpy=True)[self.realAtoms]/kilojoule_per_mole*nanometers

        # if you want energy decomposition, uncomment these lines...
        #for j in range(self.system.getNumForces()):
        #    f = self.system.getForce(j)
        #    print(type(f), str(self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
        #sys.exit()

        return np.asarray(eOPENMM), OPENMM_forces/10.0

    def compute_energy_component(self, components, energy_expression=None):
        """
        Get energy and forces from specific components

        Parameters
        -----------
        components : list
            List of strings containing names of OpenMM force classes
        energy_expression : dict
            If an OpenMM custom force class has a repeated name, then only
            get the energy and force of a class that has the specific custom expression

        Returns
        -----------
        eOPENMM : np.ndarray
            Contains OpenMM energy in kj/mol
        OPENMM_forces : np.ndarray
            Contains OpenMM force in kj/mol/A
        """
        self.simulation.step(1)
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()
        
        eOPENMM = 0
        OPENMM_forces = np.zeros_like(self.xyz_pos)
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            energy = self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
            if f.__class__.__name__ in components:
                if hasattr(f, 'getEnergyFunction') and energy_expression and f.__class__.__name__ in energy_expression and f.getEnergyFunction() == energy_expression[f.__class__.__name__]:
                    energy = self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                    eOPENMM += energy/kilojoule_per_mole
                    forces = self.simulation.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)[self.realAtoms]
                    OPENMM_forces += forces/kilojoule_per_mole*nanometers
                elif f.__class__.__name__ not in energy_expression:
                    energy = self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                    eOPENMM += energy/kilojoule_per_mole
                    forces = self.simulation.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)[self.realAtoms]
                    OPENMM_forces += forces/kilojoule_per_mole*nanometers
        return np.asarray(eOPENMM), OPENMM_forces/10.0

