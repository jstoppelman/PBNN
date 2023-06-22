from xml.etree import ElementTree
import mdtraj as md
from qm_mm import *
from nn import *
from ase.io import read, write
from ase.calculators.mixing import SumCalculator
import gc, sys
from openmm import *
from openmm.app import *

#Dictionary for converting between units, add units if you wish for more
energy_units_dict = {'kcal/mol': units.kcal/units.mol,
        'kj/mol': units.kJ/units.mol,
        'ha': units.Ha,
        'ev': units.eV}
position_units_dict = {'a': units.Angstrom,
        'bohr': units.Bohr}

def has_energy(atoms):
    """
    Returns True if ASE atoms object has atomic forces, returns False otherwise

    Parameters
    ------------
    atoms : Atoms object
        ASE atoms object

    Returns
    ------------
    bool
        Returns True if object has forces, returns False otherwise
    """
    try:
        atoms.get_potential_energy()
    except:
        return False

    return True

def has_forces(atoms):
    """
    Returns True if ASE atoms object has atomic forces, returns False otherwise

    Parameters
    ------------
    atoms : Atoms object
        ASE atoms object

    Returns
    ------------
    bool
        Returns True if object has forces, returns False otherwise
    """
    try:
        atoms.get_forces()
    except:
        return False

    return True

def get_units(position_units, energy_units, forces_units):
    """
    Converts strings for position, energy and force units into a conversion factor to kj/mol and Angstrom

    Parameters
    ------------
    position_units : str
        Units the positions in the dataset are in
    energy_units : str
        Units the energies in the dataset are in
    forces_units : str
        Units the forces are in

    Returns
    ------------
    position_conversion : float
        conversion factor to Angstrom
    energy_conversion : float
        conversion factor to kj/mol
    forces_conversion : float
        conversion factor to kj/mol/A
    """
    #Convert energy units to kJ/mol
    energy_conversion = energy_units_dict[energy_units]/energy_units_dict['kj/mol']

    if not forces_units:
        #If there isn't an explicit string for forces units, still build the conversion to kj/mol/A
        forces_conversion = (energy_units_dict[energy_units]/position_units_dict[position_units])/(energy_units_dict['kj/mol']/position_units_dict['a'])
    else:
        #Split by '/'
        unit_list = forces_units.split('/')
        #Designed to determine whether unit has multiple '/' (ex Ha/bohr vs. kJ/mol/A).
        if len(unit_list) == 2:
            eng_unit, pos_unit = forces_units.split('/')[0], forces_units.split('/')[1]
        else:
            eng_unit = unit_list[0] + '/' + unit_list[1]
            pos_unit = unit_list[2]
        forces_conversion = (energy_units_dict[eng_unit]/position_units_dict[pos_unit])/(energy_units_dict['kj/mol']/position_units_dict['a'])

    position_conversion = (position_units_dict[position_units])
    return position_conversion, energy_conversion, forces_conversion

def get_data(db_files, shift, energy_units, forces_units, data_stride):
    """
    ASE databases and ASE trajectories need to be read differently. 
    This function determines which type the file is and then calls the 
    appropriate function to get the data.

    Parameters
    -----------
    db_files : list
        list of file names that the dataset is contained in
    shift : float
        shift applied to the energy values in the dataset. 
        (This is usually the relaxed electronic energies of the 
        monomers within a dimer if these haven't already been 
        subtracted out)
    energy_units : str
        conversion factor converting the energy to kj/mol
    forces_units : str 
        conversion factor converting the forces to kj/mol/A
    data_stride : int
        read every n frames of the trajectory

    Returns
    -----------
    data : list
        List of ASE atoms objects
    energy : list
        List of data point energies
    forces : list
        List of data point forces
    e_potential : list 
        List of the external electrical potential on each atom for each data point
    e_field : list
        List of the external field on each atom for each data point
    """
    #Note some of these could be empty lists if not present in the dataset
    data, energy, forces, e_potential, e_field = [], [], [], [], []

    #Loop through listed db_files and try to read in using ASE database
    #Otherwise read in ASE trajectory
    for i, f in enumerate(db_files):
       try:
           database = db.connect(f)
           data_db, energy, forces, e_potential, e_field = read_ase_db(database,
                   energy,
                   forces,
                   e_potential,
                   e_field,
                   shift,
                   energy_units=energy_units,
                   forces_units=forces_units,
                   stride=data_stride[i])

           for d in data_db: data.append(d)
           data_db = None

       except:
           data_db, energy, forces = read_ase(f, energy, forces, shift, energy_units=energy_units, forces_units=forces_units, stride=data_stride[i])
           for d_db in data_db: data.append(d_db)
           for i in range(len(data_db)):
               e_field.append([])
               e_potential.append([])

           data_db = None

    return data, energy, forces, e_potential, e_field

def read_ase(data, 
        energy, 
        forces, 
        shift, 
        energy_units, 
        position_units="a", 
        forces_units=None, 
        stride=1):
    """
    Reads the ASE datafile and returns list of Atoms objects, energies and forces if available

    Parameters
    -----------
    data : str
        Filename containing the training data
    energy : list
        List to which the energy for each data point will be appended
    forces : list
        List to which the forces for each data point will be appended
    shift : float
        shift (default in Ha) for subtracting the gas phase electronic energy of the isolated monomers
    energy_units : str
        denotes the units the energy is in
    position_units : str
        denotes units the positions are in
    forces_units : str
        denotes the units the forces are in
    stride : int
        used for loading every nth sample from the data input file

    Returns
    -----------
    data : list
        List of ASE atoms objects
    energy : list
        array containing the energies from the dataset
    forces : list
        returns forces from the dataset. Will be empty list if no forces are present
    """
    #Load data trajectory
    data = read(data, index=f"::{stride}")

    position_conversion, energy_conversion, forces_conversion = get_units(position_units, energy_units, forces_units) 
    for i, frame in enumerate(data):

        #Ensures dataset contains the same properties for all frames
        if not has_forces(frame) and len(forces):
            raise Exception(f"Atoms object {i} missing forces while others Atoms objects in training set have it")

        elif has_forces(frame):
            force = frame.get_forces()
            force *= forces_conversion
            forces.append(force)
        
        if not has_energy(frame) and len(energy):
            raise Exception(f"Atoms object {i} missing energy while others Atoms objects in training set have it")
        
        elif has_energy(frame):
            eng = frame.get_potential_energy() - shift
            eng *= energy_conversion
            energy.append(eng)

        frame.positions *= position_conversion

    return data, energy, forces

def read_ase_db(database, 
        energy,
        forces,
        e_potential,
        e_field,
        shift, 
        energy_units, 
        position_units="a", 
        forces_units=None, 
        stride=1):
    """
    Reads the ASE database and returns list of Atoms objects, energies and forces if available.
    The database also has additional properties, such as the electrical potential or field.

    Parameters
    -----------
    database : str
        Filename containing the training data
    energy : list
        List to which the energy for each data point will be appended
    forces : list
        List to which the forces for each data point will be appended
    e_potential : list
        List to which the e_potential data will be appended if present in the database
    e_field : list
        List to which the e_field data will be appended if present in the database
    shift : float
        shift (default in Ha) for subtracting the gas phase electronic energy of the isolated monomers
    energy_units : str
        denotes the units the energy is in
    position_units : str
        denotes units the positions are in
    forces_units : str
        denotes the units the forces are in
    stride : int
        used for loading every nth sample from the data input file

    Returns
    -----------
    data : list
        List of ASE atoms objects
    energy : list
        array containing the energies from the dataset
    forces : list
        returns forces from the dataset. Will be empty list if no forces are present
    e_potential : list
        set of potentials from solvent atoms on reacting complex atoms if present in the dataset
    e_field : list
        set of fields from solvent atoms on reacting complex atoms if present in the dataset
    """
    position_conversion, energy_conversion, forces_conversion = get_units(position_units, energy_units, forces_units)
    
    data = []
    #Loop through database
    for i in range(1, len(database)+1, stride):
        atoms = database.get_atoms(id=i)
        data.append(atoms)

        if not has_energy(atoms) and len(energy):
            raise Exception(f"Atoms object {i} missing energy while others Atoms objects in training set have it")
        elif has_energy(atoms):
            eng = atoms.get_potential_energy() - shift
            eng *= energy_conversion
            energy.append(eng)

        if not has_forces(atoms) and len(forces):
            raise Exception(f"Atoms object {i} missing forces while others Atoms objects in training set have it")

        elif has_forces(atoms):
            force = atoms.get_forces()
            force *= forces_conversion
            forces.append(force)
        
        row = database.get(id=i)
        potential = row.data["e_potential"]
        e_potential.append(potential)
        field = row.data["e_field"]
        e_field.append(field)

    return data, energy, forces, e_potential, e_field

class QMMMSetup:
    """
    Read xml input settings and set up ASE simulation object
    """
    def __init__(self, settings):
        """
        settings : dictionary
            contains dictionary of settings read from xml file
        """
        self.settings = settings

        #Settings related to OpenMM
        self._mm_settings()
        #Settings related to Psi4
        self._qm_settings()
        #Settings from Plumed
        if "Plumed_Settings" in self.settings.keys():
            self._plumed_settings()
        else:
            self.plumed_command = []
        #General simulation settings
        self._simulation_settings()
        self._database_settings()

    def _mm_settings(self):
        ################ Block for settings for OpenMM Interface ################
        omm_settings = self.settings["OpenMM_Settings"]
        #directory where force field files are stored
        ffdir = omm_settings["ffdir"]
        #Get all settings for OpenMMInterface
        self.pdb_file, residue_xml_file, ff_xml_file, platform, self.qm_atoms = omm_settings["pdb_file"], omm_settings["res_file"], omm_settings["ff_file"], omm_settings["platform"], omm_settings["qm_atoms"]

        #splits string in order to form a list, as you may want to load more than one xml file
        self.residue_xml_list = residue_xml_file.split(',')
        self.residue_xml_list = [ffdir+'/'+res_file for res_file in self.residue_xml_list]

        ff_xml_list = ff_xml_file.split(',')
        ff_xml_list = [ffdir +'/'+ff_file for ff_file in ff_xml_list]

        self.mm = OpenMMInterface(ffdir+'/'+self.pdb_file, self.residue_xml_list, ff_xml_list, platform, self.qm_atoms)

    def _qm_settings(self):
        ################ Block for settings for Psi4 Interface ################

        psi4_settings = self.settings["Psi4_Settings"]

        functional, basis_set = psi4_settings["dft_functional"], psi4_settings["basis_set"]
        quadrature_radial, quadrature_spherical, qm_charge, qm_spin = psi4_settings["quadrature_radial"], psi4_settings["quadrature_spherical"], psi4_settings["qm_charge"], psi4_settings["qm_spin"]
        quadrature_radial, quadrature_spherical, qm_charge, qm_spin = int(quadrature_radial), int(quadrature_spherical), int(qm_charge), int(qm_spin)
        n_threads = int(psi4_settings["n_threads"])
        read_guess = psi4_settings["read_guess"] == "True"
        pruning, self.embedding_cutoff = psi4_settings["pruning"], int(psi4_settings["embedding_cutoff"])

        self.qm = Psi4Interface(basis_set, functional, quadrature_spherical, quadrature_radial, qm_charge, qm_spin,
                n_threads, read_guess=read_guess, pruning=pruning, SOSCF=False)

    def _plumed_settings(self):
        ################ Block for settings for Plumed ################

        plumed_settings = self.settings["Plumed_Settings"]

        plumed_file = plumed_settings["plumed_file"]
        self.plumed_command = []
        for line in open(plumed_file):
            self.plumed_command.append(line.strip('\n'))

    def _simulation_settings(self):
        ################ Block for general simulation settings ################

        simulation_settings = self.settings["Simulation_Settings"]

        self.simulation_settings_formatted = {}
        self.tmp_dir = simulation_settings["tmp_dir"]
        self.name = simulation_settings["name"]
        self.jobtype = simulation_settings.get("jobtype", None)
        if self.jobtype == "single_point":
            self.simulation_settings_formatted["jobtype"] = self.jobtype
            stride = simulation_settings["stride"]
            start = simulation_settings.get("start_idx", 0)
            self.atoms = read(simulation_settings["atoms"], index=f"{start}::{stride}")
            self.simulation_settings_formatted["atoms"] = self.atoms
            self.simulation_settings_formatted["name"] = self.name
            self.rewrite = True
        else:
            time_step, num_steps = float(simulation_settings["time_step"]), int(simulation_settings["num_steps"])
            temp, temp_init = float(simulation_settings["temp"]), float(simulation_settings["temp_init"])
            remove_rotation, remove_translation = simulation_settings["remove_rotation"] == "True", simulation_settings["remove_translation"] == "True"
            #Frequency to write output files
            write_freq = int(simulation_settings["write_freq"])
            self.atoms = simulation_settings["atoms"]
            self.restart = simulation_settings.get("restart", "False")
            self.restart = self.restart == "True"
            self.rewrite = False if self.restart else True
            self.simulation_settings_formatted["name"] = self.name
            self.simulation_settings_formatted["ensemble"] = simulation_settings["ensemble"]
            self.simulation_settings_formatted["time_step"] = time_step
            self.simulation_settings_formatted["num_steps"] = num_steps
            self.simulation_settings_formatted["temp"] = temp
            self.simulation_settings_formatted["temp_init"] = temp_init
            self.simulation_settings_formatted["remove_rotation"] = remove_rotation
            self.simulation_settings_formatted["remove_translation"] = remove_translation
            self.simulation_settings_formatted["write_freq"] = write_freq
            self.simulation_settings_formatted["restart"] = self.restart

    def _database_settings(self):
        ############## Block for storing structures and other data during a QM/MM MD simulation ##########
        database_settings = self.settings.get("NNDB_Settings", None)
        if database_settings:
            ffdir = database_settings["ffdir"]
            #Used for the reordering graph
            pdb_files = database_settings["diabat_pdbs"]
            save_frq = int(database_settings["save_frq"])
            #Atoms that are reacting in each diabat
            reacting_atoms = database_settings["reacting_atom_index"]
            #Atoms that accept the reacting atoms in diabat 2
            accepting_atoms = database_settings["accepting_atom_index"][1]
            #Load in OpenMM bond definitions to get diabat topology
            for f in self.residue_xml_list: Topology.loadBondDefinitions(f)
            omm_topology = [PDBFile(pdb_files[0]).topology, PDBFile(pdb_files[1]).topology]
            #Write or append mode
            mode = database_settings["write_mode"]
            #Residues in the diabat
            diabat_residues = database_settings.get("diabat_residues", None)[1]

            graph_reorder = GraphReorder(omm_topology, reacting_atoms, accepting_atoms, diabat_residues=diabat_residues)
            #Class that stores data during QM/MM simulation
            self.db_builder = NNDataBaseBuilder(pdb_files, graph_reorder, self.qm_atoms, self.tmp_dir, save_frq, mode)
        
        else:
            self.db_builder = None

    def get_qmmm(self):
        """
        Returns QMMM environment class

        Returns
        -----------
        self.qmmm : class
            QMMM environment class
        self.simulation_settings_formatted : dict
            Dictionary with the QMMM settings contained
        """
        #Instantiate the QM/MM system object.
        self.qmmm = QMMMEnvironment(self.atoms, self.tmp_dir, self.mm, self.qm, self.qm_atoms, self.embedding_cutoff, plumed_command=self.plumed_command, dataset_builder=self.db_builder, rewrite_log=self.rewrite)
        return self.qmmm, self.simulation_settings_formatted

class TrainSetup:
    """
    Set up neural network training
    """
    def __init__(self, settings):
        """
        settings : dict
            dictionary of settings
        """
        self.settings = settings

        self.train_settings = self.settings["Train_Settings"]

        #Get train type and settings
        train_type = self.train_settings["train_type"]
        self.train_name = self.train_settings["train_name"]
        self.train_db = self.train_settings["train_db"]
        self.train_dir = self.train_settings["train_dir"]
        data_stride = self.train_settings["data_stride"].split(',')
        self.data_stride = [int(i) for i in data_stride]
  
        #Determine whether to rebuild database or use a currently constructed database
        self.use_current_db = self.train_settings["use_current_db"] == "True"
        #Determine whether to reload existing checkpoint
        self.continue_train = self.train_settings["continue_train"] == "True"

        #Setup different training types
        if train_type == "intra":
            self.setup_intra()
        elif train_type == "inter":
            self.setup_inter()
        #Coupling is for when the intermolecular neural networks (diabats) are already determined
        elif train_type == "coupling":
            self.setup_coupling()
        #qmmm_fit fits to QM/MM data and optimizes the coupling and hii inter neural networks 
        elif train_type == "qmmm_fit":
            self.setup_train_qmmm()

    def setup_intra(self):

        #Determine whether to subtract OpenMM Morse potential energy and force from the reference data
        if "OpenMM_Settings" in self.settings.keys():
            omm_settings = self.settings["OpenMM_Settings"]

            ffdir = omm_settings['ffdir']
            pdb_file = ffdir + '/' + omm_settings['pdb_file']
            ff_file = ffdir + '/' + omm_settings['ff_file']
            res_file = ffdir + '/' + omm_settings['res_file']
            platform = omm_settings['platform']
            #If drude oscillators are present in the system, then use the Drude Hyper force
            if "drude_hyper_force" in omm_settings.keys(): drude_hyper_force = omm_settings["drude_hyper_force"] == "True"
            else: drude_hyper_force = False
            #Components is a list of energy components in OpenMM to get
            #Energy expression is a dictionary containing the energy component name with the energy expression used for the 
            #Custom class. Ex. there can be a CustomBondForce corresponding to the Morse potential and a CustomBondForce used
            #for the Drude hyper force and we only want the CustomBondForce corresponding to the Morse potential
            kwargs = {'components': ['CustomBondForce'], 'energy_expression': {'CustomBondForce': 'D*(1 - exp(-a*(r - r0)))^2'}}
            #exclude_monomer_intra can be set to [0] as it should be the only residue in the NNIntra case
            openmm = OpenMM_ForceField(pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, exclude_intra_res=[0], **kwargs)

        else:
            openmm = None

        #Fermi-Dirac function damping setup for the NN
        if "FD_Damping_Settings" in self.settings.keys():
            fd_settings = self.settings["FD_Damping_Settings"]

            mu = float(fd_settings["mu"])
            beta = float(fd_settings['beta'])
            reacting_atom_parent = fd_settings['reacting_atom_parent']
            reacting_atom_dissoc = fd_settings['reacting_atom_dissoc']

        else:
            mu = None
            beta = None
            reacting_atom_parent = None
            reacting_atom_dissoc = None

        #Get energy and force units and check they are of the accepted types
        energy_units = self.train_settings["energy_units"].lower()
        forces_units = self.train_settings["forces_units"].lower()
        if energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        #Subtract the ground state minimum energy from the training data
        shift = float(self.train_settings["shift"])

        #Either PaiNN or SchNet neural network representation
        representation = self.train_settings.get("representation", "PaiNN")
      
        db_files = self.train_db.split(',')
        #Read database
        data, energy, forces, e_potential, e_field = get_data(db_files, 
                shift, 
                energy_units=energy_units, 
                forces_units=forces_units, 
                data_stride=self.data_stride
                )
        
        #Train_NNIntra object
        nnintra_train = Train_NNIntra(
            data,
            energy,
            forces,
            openmm=openmm,
            name=self.train_name,
            use_current_db=self.use_current_db,
            continue_train=self.continue_train
            )
    
        nnintra_train.construct_model(
                representation=representation,
                mu=mu,
                beta=beta,
                reacting_atom_parent=reacting_atom_parent,
                reacting_atom_dissoc=reacting_atom_dissoc,
                train_dir=self.train_dir
                )

        self.trainer = nnintra_train

    def setup_inter(self):
        #Get OpenMM forcefield object
        if "OpenMM_Settings" in self.settings.keys():
            omm_settings = self.settings["OpenMM_Settings"]

            ff_dir = omm_settings["ffdir"]
            pdb_file = ff_dir + "/" + omm_settings['pdb_file']
            ff_file = ff_dir + "/" + omm_settings['ff_file']
            res_file = ff_dir + "/" + omm_settings['res_file']
            platform = omm_settings['platform']
            cutoff = omm_settings.get('cutoff', None)
            drude_hyper_force = omm_settings["drude_hyper_force"] == "True"
            pbnn_res = openmm_settings["pbnn_res"].split(',')
            pbnn_res = [int(res) for res in pbnn_res]

            openmm = OpenMM_ForceField(pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, cutoff=cutoff, exclude_monomer_intra=pbnn_res)

        else:
            openmm = None

        #Get Fermi-Dirac damping function settings
        if "FD_Damping_Settings" in self.settings.keys():
            fd_settings = self.settings["FD_Damping_Settings"]

            mu = float(fd_settings["mu"])
            beta = float(fd_settings['beta'])
            reacting_atom_parent = fd_settings['reacting_atom_parent']
            reacting_atom_dissoc = fd_settings['reacting_atom_dissoc']

        else:
            mu = None
            beta = None
            reacting_atom_parent = None
            reacting_atom_dissoc = None

        #Check energy and force units are accepted
        energy_units = self.train_settings["energy_units"].lower()
        if "forces_units" in self.train_settings.keys():
            forces_units = self.train_settings["forces_units"].lower()
        else:
            forces_units = None
        if energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        #Shift to subtract from the database
        shift = float(self.train_settings["shift"])
        data, energy, forces, e_potential, e_field = get_data(self.train_db, shift, energy_units=energy_units, forces_units=forces_units, stride=self.data_stride)

        #Train_NNInter object
        nninter_train = Train_NNInter(
            data,
            energy,
            forces,
            openmm=openmm,
            name=self.train_name,
            use_current_db=self.use_current_db,
            continue_train=self.continue_train
            )

        nninter_train.construct_model(mu=mu,
                beta=beta,
                reacting_atom_parent=reacting_atom_parent,
                reacting_atom_dissoc=reacting_atom_dissoc,
                train_dir=self.train_dir,
                )
        
        self.trainer = nninter_train

    def setup_coupling(self):
        openmm_settings = self.settings["OpenMM_Settings"]
        ff_dir = openmm_settings["ffdir"]
        ff_file = ff_dir + "/" + openmm_settings['ff_file']
        res_file = ff_dir + "/" + openmm_settings['res_file']
        platform = openmm_settings['platform']
        cutoff = openmm_settings.get('cutoff', None)
        pbnn_res = openmm_settings["pbnn_res"].split(',')
        pbnn_res = [int(res) for res in pbnn_res]

        drude_hyper_force = openmm_settings["drude_hyper_force"] == "True"

        energy_units = self.train_settings["energy_units"].lower()
        forces_units = self.train_settings["forces_units"].lower()

        if energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        diabat_settings = self.settings["Diabats"]
        shift = float(diabat_settings[0]["Shift"]["shift"])
        data_stride = int(self.train_settings["data_stride"])

        #Note for now the e_potential and e_field won't be used if you call this coupling training type
        data, energy, forces, e_potential, e_field = get_data(self.train_db, shift, energy_units=energy_units, forces_units=forces_units, stride=self.data_stride)

        diabats = []
        #The xml file should contain settings for two diabats. Load the settings for each diabat
        for i, diabat_setting in enumerate(diabat_settings):
            pdb_file = diabat_setting["PDB_File"]["pdb_file"]
            pdb_file = ff_dir + "/" + pdb_file
            openmm = OpenMM_ForceField(pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, cutoff=cutoff, exclude_monomer_intra=pbnn_res)

            #Use MDTraj to get the indices of atoms in each dimer
            traj = md.load(pdb_file)
            res_indices = []
            res_elements = []
            for j, res in enumerate(traj.top.residues):
                index = []
                elements = []
                for atom in res.atoms:
                    index.append(atom.index)
                    elements.append(atom.element.number)
                res_indices.append(index)
                res_elements.append(elements)

            #Set up the already trained intra neural network
            nnintra_settings = diabat_setting["NNIntra_Settings"]
            for nn in nnintra_settings:
                residue = int(nn["residue"])
                nn["atom_index"] = res_indices[residue]

            #Add additional settings corresponding to atom elements to the inter neural network settings
            nninter_settings = diabat_setting["NNInter_Settings"]
            nninter_settings["indices"] = res_indices
            nninter_settings["ZA"] = res_elements[0]
            nninter_settings["ZB"] = res_elements[1]

            #If this is the first diabat, then we don't need to reorder the positions. If not, then we need to 
            #set up the GraphReorder class which reorders the positions to another topology
            if i == 0:
                diabat = Diabat(openmm, nnintra_settings, nninter_settings)
            else:
                pdb_files = [ff_dir+"/"+diabat_settings[0]["PDB_File"]["pdb_file"], ff_dir+"/"+diabat_settings[1]["PDB_File"]["pdb_file"]]
                
                reacting_atoms = [diabat_settings[0]["Graph_Settings"]["reacting_atom_index"], diabat_settings[1]["Graph_Settings"]["reacting_atom_index"]]
                accepting_atoms = diabat_settings[1]["Graph_Settings"]["accepting_atom_index"]

                omm_topology = [diabats[0].openmm.pdb.topology, openmm.pdb.topology]
                diabat_residues = diabat_settings[1]["Graph_Settings"].get("diabat_residues", None)
                reorder_graph = GraphReorder(omm_topology, reacting_atoms, accepting_atoms, diabat_residues=diabat_residues)

                shift = float(diabat_settings[i]["Shift"]["shift"]) * energy_units_dict[energy_units]/energy_units_dict["kj/mol"]
                diabat = Diabat(openmm, nnintra_settings, nninter_settings, reorder_graph=reorder_graph, shift=shift)

            diabats.append(diabat)

        nn_hij = Train_NNHij(
                    data,
                    energy,
                    forces,
                    diabats,
                    name=self.train_name,
                    use_current_db=self.use_current_db,
                    continue_train=self.continue_train
                )

        #Set damping functions for reactant and product
        reacting_atom_parent_react = diabat_settings[0]["FD_Damping_Settings"]["reacting_atom_parent"]
        reacting_atom_dissoc_react = diabat_settings[0]["FD_Damping_Settings"]["reacting_atom_dissoc"]
        reacting_atom_parent_prod = diabat_settings[1]["FD_Damping_Settings"]["reacting_atom_parent"]
        reacting_atom_dissoc_prod = diabat_settings[1]["FD_Damping_Settings"]["reacting_atom_dissoc"]
        mu = float(diabat_settings[0]["FD_Damping_Settings"]["mu"])
        beta = float(diabat_settings[0]["FD_Damping_Settings"]["beta"])
        mu_prod = float(diabat_settings[1]["FD_Damping_Settings"]["mu"])
        beta_prod = float(diabat_settings[1]["FD_Damping_Settings"]["beta"])

        nn_hij.construct_model(
                reacting_atom_parent_react=reacting_atom_parent_react,
                reacting_atom_dissoc_react=reacting_atom_dissoc_react,
                reacting_atom_parent_prod=reacting_atom_parent_prod,
                reacting_atom_dissoc_prod=reacting_atom_dissoc_prod,
                mu_react=mu,
                beta_react=beta,
                mu_prod=mu_prod,
                beta_prod=beta_prod,
                train_dir=self.train_dir
                )

        self.trainer = nn_hij

    def setup_train_qmmm(self):
        openmm_settings = self.settings["OpenMM_Settings"]
        ff_dir = openmm_settings["ffdir"]
        ff_file = ff_dir + "/" + openmm_settings['ff_file']
        res_file = ff_dir + "/" + openmm_settings['res_file']
        platform = openmm_settings['platform']
        cutoff = openmm_settings.get('cutoff', None)
        pbnn_res = openmm_settings["pbnn_res"].split(',')
        pbnn_res = [int(res) for res in pbnn_res]

        if "drude_hyper_force" in openmm_settings.keys():
            drude_hyper_force = openmm_settings["drude_hyper_force"] == "True"
        else:
            drude_hyper_force = False

        energy_units = self.train_settings["energy_units"].lower()
        forces_units = self.train_settings["forces_units"].lower()

        if energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        diabat_settings = self.settings["Diabats"]
        shift = float(diabat_settings[0]["Shift"]["shift"])

        db_files = self.train_db.split(',')
        data, energy, forces, e_potential, e_field = get_data(db_files, shift, energy_units, forces_units, self.data_stride)
        
        energy = np.asarray(energy)
        forces = np.asarray(forces)
        
        #Loop through and get diabat settings
        diabats = []
        for i, diabat_setting in enumerate(diabat_settings):
            pdb_file = diabat_setting["PDB_File"]["pdb_file"]
            pdb_file = ff_dir + "/" + pdb_file
            openmm = OpenMM_ForceField(pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, cutoff=cutoff, exclude_intra_res=pbnn_res)

            #Use MDTraj to get the indices of atoms in each dimer
            traj = md.load(pdb_file)
            res_indices = []
            res_elements = []
            for j, res in enumerate(traj.top.residues):
                index = []
                elements = []
                for atom in res.atoms:
                    index.append(atom.index)
                    elements.append(atom.element.number)
                res_indices.append(index)
                res_elements.append(elements)

            #Get NN intra settings
            nnintra_settings = diabat_setting["NNIntra_Settings"]
            for nn in nnintra_settings:
                residue = int(nn["residue"])
                nn["atom_index"] = res_indices[residue]

            #Don't need to set up NN Inter here since this is being trained
            nninter_settings = {}

            #If this is the first diabat, then we don't need to reorder the positions. If not, then we need to 
            #set up the GraphReorder class which reorders the positions to another topology
            if i == 0:
                diabat = Diabat(openmm, nnintra_settings, nninter_settings)
            else:
                pdb_files = [ff_dir+"/"+diabat_settings[0]["PDB_File"]["pdb_file"], ff_dir+"/"+diabat_settings[1]["PDB_File"]["pdb_file"]]
                reacting_atoms = [diabat_settings[0]["Graph_Settings"]["reacting_atom_index"], diabat_settings[1]["Graph_Settings"]["reacting_atom_index"]]
                accepting_atoms = diabat_settings[1]["Graph_Settings"]["accepting_atom_index"]

                omm_topology = [diabats[0].openmm.pdb.topology, openmm.pdb.topology]
                diabat_residues = diabat_settings[1]["Graph_Settings"].get("diabat_residues", None)
                reorder_graph = GraphReorder(omm_topology, reacting_atoms, accepting_atoms, diabat_residues=diabat_residues)

                shift = float(diabat_settings[i]["Shift"]["shift"]) * energy_units_dict[energy_units]/energy_units_dict["kj/mol"]
                diabat = Diabat(openmm, nnintra_settings, nninter_settings, reorder_graph=reorder_graph, shift=shift)

            diabats.append(diabat)

        nn_qmmm = Train_NNQMMM(
                    data,
                    energy,
                    forces,
                    diabats,
                    e_field=e_field,
                    e_potential=e_potential,
                    name=self.train_name,
                    use_current_db=self.use_current_db,
                    continue_train=self.continue_train
                )

        #Setup inter NN for each diabat
        for i, diabat_setting in enumerate(diabat_settings):

            mu = float(diabat_settings[i]["FD_Damping_Settings"]["mu"])
            beta = float(diabat_settings[i]["FD_Damping_Settings"]["beta"])
            reacting_atom_parent = diabat_settings[i]["FD_Damping_Settings"]["reacting_atom_parent"]
            reacting_atom_dissoc = diabat_settings[i]["FD_Damping_Settings"]["reacting_atom_dissoc"]

            nn_qmmm.construct_inter_model(mu=mu,
                    beta=beta,
                    reacting_atom_parent=reacting_atom_parent,
                    reacting_atom_dissoc=reacting_atom_dissoc,
                    diabat=i,
                    )

        #Hij damping functions for reactand and product
        mu = float(diabat_settings[0]["FD_Damping_Settings"]["mu"])
        beta = float(diabat_settings[0]["FD_Damping_Settings"]["beta"])
        reacting_atom_parent = diabat_settings[0]["FD_Damping_Settings"]["reacting_atom_parent"]
        reacting_atom_dissoc = diabat_settings[0]["FD_Damping_Settings"]["reacting_atom_dissoc"]

        mu_prod = float(diabat_settings[1]["FD_Damping_Settings"]["mu"])
        beta_prod = float(diabat_settings[1]["FD_Damping_Settings"]["beta"])
        reacting_atom_parent_prod = diabat_settings[1]["FD_Damping_Settings_Coupling"]["reacting_atom_parent"]
        reacting_atom_dissoc_prod = diabat_settings[1]["FD_Damping_Settings_Coupling"]["reacting_atom_dissoc"]
        
        nn_qmmm.construct_hij_model(
                mu_react=mu,
                beta_react=beta,
                mu_prod=mu_prod,
                beta_prod=beta_prod,
                reacting_atom_parent_react=reacting_atom_parent,
                reacting_atom_dissoc_react=reacting_atom_dissoc,
                reacting_atom_parent_prod=reacting_atom_parent_prod,
                reacting_atom_dissoc_prod=reacting_atom_dissoc_prod,
                )
        
        nn_qmmm.construct_database(train_dir=self.train_dir)

        self.trainer = nn_qmmm

    def get_trainer(self):
        """
        Returns
        --------
        self.trainer : class
            Class used to train a particular model type
        """
        return self.trainer

class PBNNSetup:
    def __init__(self, settings):
        """
        Setup PBNN class either for single point tests or MD simulation

        Parameters
        -----------
        settings : dict
            Dictionary containing various settings
        """

        self.settings = settings

        #Settings specific to job or simulation
        self._get_pbnn_settings()
        self.diabats = self._get_diabat_settings()
        self.coupling_settings = self._get_coupling_settings()
        self.plumed_command = self._get_plumed_settings()

    def _get_pbnn_settings(self):
        
        self.pbnn_settings = self.settings["PBNN_Settings"]

        #Test runs singlepoint energy calculations on a dataset
        self.jobtype = self.pbnn_settings["jobtype"]
        if self.jobtype == "Test":
            self.data_stride = self.pbnn_settings["data_stride"].split(',')
            self.data_stride = [int(i) for i in self.data_stride]
            #Indices for a specific subset of data in the database file
            #Otherwise goes through the whole supplied datafile
            self.test_idx = self.pbnn_settings.get("test_idx", None)
            if self.test_idx:
                self.test_idx = np.load(self.test_idx)["test_idx"]
        #Either dataset for Test jobtype or Atoms object for MD jobtype
        self.atoms = self.pbnn_settings["atoms"]
        self.energy_units = self.pbnn_settings["energy_units"].lower()
        self.forces_units = self.pbnn_settings["forces_units"].lower()

        shift = float(self.settings["Diabats"][0]["Shift"]["shift"])

        if self.energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        if self.jobtype == "Test":
            db_files = self.atoms
            db_files = db_files.split(',')

            self.data, self.energy, self.forces, self.e_potential, self.e_field = get_data(db_files, shift, self.energy_units, self.forces_units, self.data_stride)

        elif self.jobtype == "MD":
            self.data = self.atoms

    def _get_diabat_settings(self):
        """
        Returns
        ----------
        diabats : list
            List of Diabat classes
        """
        #Settings specific to a given diabat

        openmm_settings = self.settings["OpenMM_Settings"]
        self.ff_dir = openmm_settings["ffdir"]
        ff_file = self.ff_dir + "/" + openmm_settings['ff_file']
        res_file = self.ff_dir + "/" + openmm_settings['res_file']
        platform = openmm_settings['platform']
        cutoff = openmm_settings.get("cutoff", None)
        pbnn_res = openmm_settings["pbnn_res"].split(',')
        pbnn_res = [int(res) for res in pbnn_res]
        if "drude_hyper_force" in openmm_settings.keys():
            drude_hyper_force = openmm_settings["drude_hyper_force"] == "True"
        else:
            drude_hyper_force = False

        diabat_settings = self.settings["Diabats"]
        diabats = []
        #Loop through and create settings for each diabat
        for i, diabat_setting in enumerate(diabat_settings):
            pdb_file = diabat_setting["PDB_File"]["pdb_file"]
            pdb_file = self.ff_dir + "/" + pdb_file
            react_atom = int(diabat_setting["React_Atom"]["atom_index"])
            react_residue = int(diabat_setting["React_Atom"]["residue"])
            openmm = OpenMM_ForceField(pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, cutoff=cutoff, exclude_intra_res=pbnn_res)
            openmm.setReactAtom(react_atom, react_residue)

            #MDTraj for dimer info
            traj = md.load(pdb_file)
            res_indices = []
            res_elements = []
            for j, res in enumerate(traj.top.residues):
                index = []
                elements = []
                for atom in res.atoms:
                    index.append(atom.index)
                    elements.append(atom.element.number)
                res_indices.append(index)
                res_elements.append(elements)

            #Intra NN settings
            nnintra_settings = diabat_setting["NNIntra_Settings"]
            for nn in nnintra_settings:
                residue = int(nn["residue"])
                nn["atom_index"] = res_indices[residue]
            
            #Inter NN settings
            nninter_settings = diabat_setting["NNInter_Settings"]
            residues = nninter_settings["residue"]
            residues = residues.split(',')
            residues = [int(res) for res in residues]
            nn_residues = [res_indices[res] for res in residues]
            nninter_settings["indices"] = nn_residues
            nninter_settings["ZA"] = res_elements[residues[0]]
            nninter_settings["ZB"] = res_elements[residues[1]]
            
            #As before, only assemble GraphReorder class if the diabat isn't the initial one
            if i == 0:
                diabat = Diabat(openmm, nnintra_settings, nninter_settings)
            else:
                pdb_files = [self.ff_dir+"/"+diabat_settings[0]["PDB_File"]["pdb_file"], self.ff_dir+"/"+diabat_settings[1]["PDB_File"]["pdb_file"]]
                reacting_atoms = [diabat_settings[0]["Graph_Settings"]["reacting_atom_index"], diabat_settings[1]["Graph_Settings"]["reacting_atom_index"]]
                accepting_atoms = diabat_settings[1]["Graph_Settings"]["accepting_atom_index"]
                diabat_residues = diabat_settings[1]["Graph_Settings"].get("diabat_residues", None)
                omm_topology = [diabats[0].openmm.pdb.topology, openmm.pdb.topology]
                reorder_graph = GraphReorder(omm_topology, reacting_atoms, accepting_atoms, diabat_residues=diabat_residues)

                shift = float(diabat_settings[i]["Shift"]["shift"]) * energy_units_dict[self.energy_units]/energy_units_dict["kj/mol"]
                diabat = Diabat(openmm, nnintra_settings, nninter_settings, reorder_graph=reorder_graph, shift=shift)

            diabats.append(diabat)
        
        return diabats

    def _get_coupling_settings(self):
        """
        Returns
        ---------
        self.couplings : list
            List of Coupling classes
        """
        #Get settings related to couplings
        couplings = []
        self.coupling_settings = self.settings["Coupling_Settings"]

        pdb_file = self.settings["Diabats"][0]["PDB_File"]["pdb_file"]
        pdb_file = self.ff_dir + "/" + pdb_file
        traj = md.load(pdb_file)
        res_indices = []
        res_elements = []
        for j, res in enumerate(traj.top.residues):
            index = []
            elements = []
            for atom in res.atoms:
                index.append(atom.index)
                elements.append(atom.element.number)
            res_indices.append(index)
            res_elements.append(elements)

        for coupling_setting in self.coupling_settings:

            model = coupling_setting["fname"]
            couplings_loc = coupling_setting["coupling_loc"]
            couplings_loc = couplings_loc.split(',')
            couplings_loc = [int(loc) for loc in couplings_loc]
            
            residues = coupling_setting["residues"]
            residues = residues.split(',')
            residues = [int(res) for res in residues]
            indices = [index for res in residues for index in res_indices[res]]

            damping = {}
            for key, val in coupling_setting.items():
                if "damping" in key:
                    damping[key] = val
        
            coupling = Coupling(model, indices, couplings_loc, damping) 
            couplings.append(coupling)

        self.couplings = couplings

    def _get_plumed_settings(self):
        """
        Returns 
        ---------
        plumed_settings : list
            List of plumed commands
        """
        #Get Plumed Settings if present in the xml file
        plumed_settings = self.settings.get("Plumed_Settings", None)
        if plumed_settings:
            plumed_file = plumed_settings['plumed_file']
            plumed_settings = []
            #Read plumed file for commands
            for line in open(plumed_file):
                plumed_settings.append(line.strip('\n'))
        return plumed_settings

    def get_pbnn(self):
        """
        Returns
        --------
        pbnn : class
            PBNN class
        """
        #supply jobtype as keyword arg
        kwargs = {'jobtype': self.jobtype}
        tmp = self.settings["PBNN_Settings"]["tmp_dir"]
        pbnn = PBNN_Interface(self.data, self.diabats, self.couplings, tmp, plumed_command=self.plumed_command, **kwargs)
        if self.jobtype == "Test":
            #Only grab dataset indices corresponding to test_idx if present in the file
            if self.test_idx is not None:
                self.data = [self.data[i] for i in self.test_idx]
                if len(self.energy): self.energy = [self.energy[i] for i in self.test_idx]
                if len(self.forces): self.forces = [self.forces[i] for i in self.test_idx]
                if len(self.e_potential): self.e_potential = [self.e_potential[i] for i in self.test_idx]
                if len(self.e_field): self.e_field = [self.e_field[i] for i in self.test_idx]
            name = self.settings["PBNN_Settings"]["name"]
            pbnn.setup_test(
                    self.data,
                    self.energy,
                    self.forces,
                    self.e_potential,
                    self.e_field,
                    name=name
                    )
        
        elif self.jobtype == "MD":
            #Settings related to MD simulation
            name = self.settings["PBNN_Settings"]["name"]
            time_step = float(self.settings["PBNN_Settings"]["time_step"])
            temp = float(self.settings["PBNN_Settings"]["temp"])
            temp_init = float(self.settings["PBNN_Settings"]["temp_init"])
            remove_rotation = self.settings["PBNN_Settings"]["remove_rotation"] == "True"
            remove_translation = self.settings["PBNN_Settings"]["remove_translation"] == "True"
            write_frq = int(self.settings["PBNN_Settings"]["write_freq"])
            ensemble = self.settings["PBNN_Settings"]["ensemble"]
            friction = float(self.settings["PBNN_Settings"]["friction"])
            self.num_steps = int(self.settings["PBNN_Settings"]["num_steps"])

            pbnn.create_system(
                    name, 
                    time_step, 
                    temp=temp, 
                    temp_init=temp_init,
                    store=write_frq,
                    ensemble=ensemble,
                    remove_rotation=remove_rotation,
                    remove_translation=remove_translation,
                    friction=friction
                    )
        
        return pbnn

class PBNNComponent:
    """
    Designed to test the energy components of one piece of the Hamilotnian
    Like NN Intra within one diabat
    """
    def __init__(self, settings):
        """
        Parameters
        -----------
        settings : dict
            Dicitonary containing various settings
        """

        self.settings = settings
        self._get_data_settings()
        if self.jobtype == "NNIntra":
            self._get_nnintra_calc()

    def _get_data_settings(self):

        #Data settings TO DO condense this to one inherited function for each class
        self.pbnn_settings = self.settings["PBNN_Settings"]
        self.pdb_file = self.pbnn_settings["ffdir"] + '/' + self.pbnn_settings["pdb_file"]
        self.name = self.pbnn_settings["name"]
        self.jobtype = self.pbnn_settings["jobtype"]
        self.data_stride = self.pbnn_settings["data_stride"].split(',')
        self.data_stride = [int(i) for i in self.data_stride]
        self.test_idx = self.pbnn_settings.get("test_idx", None)
        if self.test_idx:
            self.test_idx = np.load(self.test_idx)["test_idx"]
        self.atoms = self.pbnn_settings["atoms"]
        self.energy_units = self.pbnn_settings["energy_units"].lower()
        self.forces_units = self.pbnn_settings["forces_units"].lower()

        shift = float(self.pbnn_settings["shift"])

        if self.energy_units not in ["kj/mol", "ha", "kcal/mol", "ev"]:
            raise ValueError(f"Unit choice {energy_units} is invalid")

        db_files = self.atoms
        db_files = db_files.split(',')

        self.data, self.energy, self.forces, self.e_potential, self.e_field = get_data(db_files, shift, self.energy_units, self.forces_units, self.data_stride)

    def _get_nnintra_calc(self):
        """
        Setup class for testing NN Intra
        """
        nnintra_settings = self.settings["NNIntra_Settings"]

        traj = md.load(self.pdb_file)
        res_indices = []
        res_elements = []
        for j, res in enumerate(traj.top.residues):
            index = []
            elements = []
            for atom in res.atoms:
                index.append(atom.index)
                elements.append(atom.element.number)
            res_indices.append(index)
            res_elements.append(elements)

        #NN Intra settings
        residue = int(nnintra_settings[0]["residue"])
        nnintra_settings[0]["atom_index"] = res_indices[residue]
        for opt in nnintra_settings:
            model = opt['fname']
            indices = opt["atom_index"]
            if "damping_parent" in opt.keys():
                parent_atom = opt["damping_parent"]
                dissoc_atom = opt["damping_dissoc"]
                parent_atom = parent_atom.split(',')
                parent_atom = [int(i) for i in parent_atom]
                if not isinstance(dissoc_atom, list): dissoc_atom = list(dissoc_atom)
                dissoc_atom = [int(i) for i in dissoc_atom]
                calc = NN_Intra(model, indices, damping=[parent_atom, dissoc_atom])
            else:
                calc = NN_Intra(model, indices)

        #Determine whether to add force field energy to the test
        omm_settings = self.settings.get("OpenMM_Settings", None)
        if omm_settings:
            kwargs = {'energy_expression': {'CustomBondForce': 'D*(1 - exp(-a*(r - r0)))^2'}, 'components': ['CustomBondForce']}
            self.ff_dir = omm_settings["ffdir"]
            ff_file = self.ff_dir + "/" + omm_settings['ff_file']
            res_file = self.ff_dir + "/" + omm_settings['res_file']
            platform = omm_settings['platform']
            cutoff = omm_settings.get("cutoff", None)
            pbnn_res = omm_settings["pbnn_res"].split(',')
            pbnn_res = [int(res) for res in pbnn_res]
            if "drude_hyper_force" in omm_settings.keys():
                drude_hyper_force = omm_settings["drude_hyper_force"] == "True"
            else:
                drude_hyper_force = False
            openmm = OpenMM_ForceField(self.pdb_file, res_file, ff_file, platformtype=platform, Drude_hyper_force=drude_hyper_force, cutoff=cutoff, exclude_intra_res=pbnn_res, **kwargs)

            #SumCalculator combines the energies and forces from separate calculators
            calc = SumCalculator([calc, openmm])
        
        self.calc = calc

    def getTest(self):
        """
        Returns
        --------
        TestComponent : class
            Class designed to test NN against a dataset
        """
        return TestComponent(self.data, self.energy, self.forces, self.calc, self.name) 

class ReadSettings:
    """
    Base ReadSettings class
    """
    def __init__(self, xml_file):
        """
        xml_file : str
            path to xml_file
        """

        self.xml_file = xml_file
        tree = ElementTree.parse(xml_file)
        self.root = tree.getroot()

    def _read_damping_settings(self, settings):
        """
        Settings specific to damping function
        Parameters
        -----------
        settings : xml etree object
            contains the set of settings for forming the Fermi-Dirac cutoff functions
        Returns
        -----------
        damping_inputs : dictionary
            contains the inputs from the settings object in order to form the Fermi-Dirac damping functions
        """

        damping_inputs = {}
        for setting in settings:
            if "reacting_atom_parent" in setting.tag or "reacting_atom_dissoc" in setting.tag:
                atoms = []
                for sub in setting:
                    atoms = sub.get("index")
                atoms = atoms.split(',')
                atoms = [int(a) for a in atoms]
                damping_inputs[setting.tag] = atoms

            if "damping_position" in setting.tag or "damping_beta" in setting.tag:
                for sub in setting:
                    damping_inputs[sub.tag] = float(sub.get("value"))
            if "react_res" in setting.tag:
                for sub in setting:
                    damping_inputs[setting.tag] = int(sub.get("index"))

        return damping_inputs

    def _read_general_settings(self, settings):
        """
        Reads through an xml block and gets the value for a specific tab
        Parameters
        -----------
        settings : xml etree object
            contains a set of settings
        Returns
        -----------
        inputs : dictionary
            contains the set of inputs from the settings object
        """
        #Loop through settings in xml Etree library
        inputs = {}
        for setting in settings:
            for key, val in setting.attrib.items():
                inputs[setting.tag] = val

        return inputs

class ReadSettingsQMMM(ReadSettings):
    """
    Read xml input for QMMM
    """
    def __init__(self, xml_file):
        """
        Parameters
        -----------
        xml_file : str
            path to xml_file
        """
        super(ReadSettingsQMMM, self).__init__(xml_file)
    
    def _read_nndb_settings(self, settings):
        """
        Reads the settings needed to build a database for NN training
        through a QM/MM simulation
        Parameters
        -----------
        settings : xml etree object
            contains a set of settings
        Returns
        -----------
        inputs : dictionary
            contains the set of inputs from the settings object

        """

        inputs = {}
        reacting_atom_indices = []
        accepting_atom_indices = []
        diabat_indices = []
        for setting in settings:
            #Get the settings for the GraphReorder class
            if setting.tag == "Graph_Settings":
                for diabat in setting:
                    for index in diabat:
                        if "reacting_atom_index" in index.tag:
                            ind = index.attrib.get("index")
                            reacting_atom_index = [int(i) for i in ind.split(',')]
                            
                        if "accepting_atom_index" in index.tag:
                            atoms = index.attrib.get("index")
                            accepting_atom_index = [int(i) for i in atoms.split(',')]

                        if "diabat_residues" in index.tag:
                            ind = index.attrib.get("index")
                            diabat_index = [int(i) for i in ind.split(',')]
                
                    reacting_atom_indices.append(reacting_atom_index)
                    accepting_atom_indices.append(accepting_atom_index)
                    diabat_indices.append(diabat_index)

            #PDB files corresponding to each diabat that will be used in the PBNN Hamiltonian
            elif setting.tag == "diabat_pdb":
                val = setting.attrib["file"]
                val = val.split(',')
                inputs["diabat_pdbs"] = val
            else:
                for key, val in setting.attrib.items():
                    inputs[setting.tag] = val

        inputs["reacting_atom_index"] = reacting_atom_indices
        inputs["accepting_atom_index"] = accepting_atom_indices
        inputs["diabat_residues"] = diabat_indices
        diabat_pdbs = inputs["diabat_pdbs"]
        ffdir = inputs["ffdir"]
        diabat_pdbs = [ffdir + '/' + pdb for pdb in diabat_pdbs]
        inputs["diabat_pdbs"] = diabat_pdbs
        return inputs

    def getSettings(self):
        """
        Returns
        -----------
        settings : dict
            dictionary of settings for starting the QMMM simulation
        """
        #stores all settings as a dictionary for each selection
        settings = {}

        #loop through ElementTree
        for child in self.root:
            if child.tag == "OpenMM_Settings":
                omm_settings = self._read_general_settings(child)
              
                ffdir = omm_settings["ffdir"]
                #From the residues labeled as the QM active site residues, get the atom indices
                traj = md.load(ffdir+"/"+omm_settings["pdb_file"])
                qm_residues = omm_settings["qm_residues"].split(',')
                atoms = []
                for res in qm_residues:
                    index = traj.top.select(f"resid {res}")
                    for i in index: atoms.append(i)
                omm_settings["qm_atoms"] = atoms
                
                settings["OpenMM_Settings"] = omm_settings

            if child.tag == "Psi4_Settings":
                psi4_settings = self._read_general_settings(child)
                settings["Psi4_Settings"] = psi4_settings

            if child.tag == "Plumed_Settings":
                plumed_settings = self._read_general_settings(child)
                settings["Plumed_Settings"] = plumed_settings

            if child.tag == "Simulation_Settings":
                simulation_settings = self._read_general_settings(child)
                settings["Simulation_Settings"] = simulation_settings

            if child.tag == "NNDB_Settings":
                nndb_settings = self._read_nndb_settings(child)
                settings["NNDB_Settings"] = nndb_settings

        return settings

class ReadSettingsTraining(ReadSettings):
    def __init__(self, xml_input):
        super(ReadSettingsTraining, self).__init__(xml_input)

    def _read_nnintra_settings(self, settings):
        """
        Parameters
        -----------
        settings : xml etree object
            contains the set of settings for forming the NN_Intra neural network
        Returns
        -----------
        nn_intra_inputs : dictionary
            contains the NN_Intra inputs from the settings object
        """

        nn_intra_inputs = []
        for setting in settings:
            nn_intra_model = {}
            for key, val in setting.attrib.items():
                nn_intra_model[key] = val
            nn_intra_inputs.append(nn_intra_model)
        return nn_intra_inputs

    def _read_nninter_settings(self, settings):
        """
        Parameters
        -----------
        settings : xml etree object
            contains the set of settings for forming the NN_Inter neural networks
        Returns
        -----------
        nn_inter_inputs : dictionary
            contains the NN_Inter inputs from the settings object
        """

        nn_inter_inputs = {}
        for setting in settings:
            for key, val in setting.attrib.items():
                nn_inter_inputs[key] = val
        return nn_inter_inputs

    def _read_graph_settings(self, settings):
        """
        Parameters
        -----------
        settings : xml etree object
            contains the set of settings for forming the reordered graphcs between diabats
        Returns
        -----------
        graph_inputs : dictionary
            contains the graph inputs from the settings object
        """

        graph_inputs = {}
        for setting in settings:
            if "reacting_atom_index" in setting.tag:
                index = setting.attrib.get("index")
                index = [int(i) for i in index.split(',')]
                graph_inputs[setting.tag] = index
            if "accepting_atom_index" in setting.tag:
                index = setting.attrib.get("index")
                index = [int(i) for i in index.split(',')]
                graph_inputs[setting.tag] = index
            if "diabat_residues" in setting.tag:
                index = setting.attrib.get("index")
                index = [int(i) for i in index.split(',')]
                graph_inputs[setting.tag] = index

        return graph_inputs

    def getSettings(self):
        """
        Returns
        -----------
        settings : dictionary
            contains the graph inputs from the settings object
        """

        settings = {}

        for child in self.root:
            #General OpenMM settings used in each diabat
            if child.tag == "OpenMM_Settings":
                omm_settings = self._read_general_settings(child)
                settings["OpenMM_Settings"] = omm_settings
            if child.tag == "Train_Settings":
                train_settings = self._read_general_settings(child)
                settings["Train_Settings"] = train_settings
            if child.tag == "Shift":
                settings["Shift"] = float(child.get("shift"))
            if child.tag == "FD_Damping_Settings":
                damping_settings = self._read_damping_settings(child)
                settings["FD_Damping_Settings"] = damping_settings
            if child.tag == "Diabat_Settings":
                diabat = {}
                if "Diabats" not in list(settings.keys()):
                    settings["Diabats"] = []
                for branch in child:
                    if branch.tag == "NNIntra_Settings":
                        nn_intra_inputs = self._read_nnintra_settings(branch)
                        diabat["NNIntra_Settings"] = nn_intra_inputs
                    if branch.tag == "NNInter_Settings":
                        nn_intra_inputs = self._read_nninter_settings(branch)
                        diabat["NNInter_Settings"] = nn_intra_inputs
                    if branch.tag == "PDB_file":
                        pdb_settings = self._read_general_settings(branch)
                        diabat["PDB_File"] = pdb_settings
                    if branch.tag == "Graph_Settings":
                        graph_settings = self._read_graph_settings(branch)
                        diabat["Graph_Settings"] = graph_settings
                    if branch.tag == "FD_Damping_Settings":
                        damping_settings = self._read_damping_settings(branch)
                        diabat["FD_Damping_Settings"] = damping_settings
                    if branch.tag == "FD_Damping_Settings_Coupling":
                        damping_settings = self._read_damping_settings(branch)
                        diabat["FD_Damping_Settings_Coupling"] = damping_settings
                    if branch.tag == "Shift":
                        shift = self._read_general_settings(branch)
                        diabat["Shift"] = shift
                settings["Diabats"].append(diabat)

        return settings

class ReadSettingsPBNN(ReadSettingsTraining):
    """
    Settings for PB/NN
    """
    def __init__(self, xml_input):
        super(ReadSettingsPBNN, self).__init__(xml_input)

    def getSettings(self):
        settings = {}

        for child in self.root:
            if child.tag == "OpenMM_Settings":
                omm_settings = self._read_general_settings(child)
                settings["OpenMM_Settings"] = omm_settings
            if child.tag == "Diabat_Settings":
                diabat = {}
                if "Diabats" not in list(settings.keys()):
                    settings["Diabats"] = []
                for branch in child:
                    if branch.tag == "NNIntra_Settings":
                        nn_intra_inputs = self._read_nnintra_settings(branch)
                        diabat["NNIntra_Settings"] = nn_intra_inputs
                    if branch.tag == "NNInter_Settings":
                        nn_inter_inputs = self._read_nninter_settings(branch)
                        diabat["NNInter_Settings"] = nn_inter_inputs
                    if branch.tag == "PDB_file":
                        pdb_settings = self._read_general_settings(branch)
                        diabat["PDB_File"] = pdb_settings
                    if branch.tag == "Graph_Settings":
                        graph_settings = self._read_graph_settings(branch)
                        diabat["Graph_Settings"] = graph_settings
                    if branch.tag == "React_Atom":
                        react_atom = self._read_general_settings(branch)
                        diabat["React_Atom"] = react_atom
                    if branch.tag == "Shift":
                        shift = self._read_general_settings(branch)
                        diabat["Shift"] = shift
                settings["Diabats"].append(diabat)
            if child.tag == "Coupling_Settings":
                coupling_settings = self._read_coupling_settings(child)
                settings["Coupling_Settings"] = coupling_settings
            if child.tag == "PBNN_Settings":
                pbnn_settings = self._read_general_settings(child)
                settings["PBNN_Settings"] = pbnn_settings
            if child.tag == "Plumed_Settings":
                plumed_settings = self._read_general_settings(child)
                settings["Plumed_Settings"] = plumed_settings

        return settings

    def _read_coupling_settings(self, settings):
        """
        Reads through an xml block and gets the value for a specific tab
        Parameters
        -----------
        settings : xml etree object
            contains a set of settings
        Returns
        -----------
        inputs : dictionary
            contains the set of inputs from the settings object
        """

        coupling_settings = []
        for setting in settings:
            coupling_setting = {}
            for key, val in setting.attrib.items():
                coupling_setting[key] = val
            coupling_settings.append(coupling_setting)
        
        return coupling_settings

class ReadSettingsPBNNComp(ReadSettingsPBNN):
    """
    Settings for the PBNN Components
    TODO Make it work for more than just NN Intra
    """
    def __init__(self, xml_input):
        super(ReadSettingsPBNNComp, self).__init__(xml_input)

    def getSettings(self):
        settings = {}
        for child in self.root:
            if child.tag == "OpenMM_Settings":
                omm_settings = self._read_general_settings(child)
                settings["OpenMM_Settings"] = omm_settings
            if child.tag == "NNIntra_Settings":
                nn_intra_inputs = self._read_nnintra_settings(child)
                settings["NNIntra_Settings"] = nn_intra_inputs
            if child.tag == "PBNN_Settings":
                pbnn_settings = self._read_general_settings(child)
                settings["PBNN_Settings"] = pbnn_settings
        return settings


