import numpy as np
import os, sys, shutil
from schnetpack.data import ASEAtomsData, AtomsDataModule
from schnetpack.transform.fd_setup import FDSetup, FDSetup_OffDiag, FD_Simultaneous, FDSetup_SchNet
from schnetpack.transform.neighborlist import (TorchNeighborList, 
        APNetNeighborList, 
        APOffDiagNeighborList,
        APNetPBCNeighborList,
        APOffDiagPBCNeighborList,
        APSimultaneousNeighborList)
from schnetpack.transform.casting import CastTo32
from schnetpack.nn.radial import GaussianRBF
from schnetpack.train.callbacks import ModelCheckpoint
import schnetpack as sch
import torchmetrics
import torch
from torch.optim import Adam
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger
from .diabat import reorder
from .custom_task import CustomTask
import pickle

class Train_NN:
    """
    Base class to set up training for all neural networks
    """
    def __init__(self, data, energy, forces, name=None, use_current_db=False, delete_data=True, continue_train=False, add_data=False):
        """
        data : list
            list of ASE atoms objects, represents the dataset
        energy : np.ndarray
            contains the energy of the different data points
        forces : np.ndarray
            contains the forces for each data point
        name : str or None
            (optional) can be used to name the SchNet AtomsData database
        use_current_db : bool
            if there is a current db, this determines whether to leave it as is or delete it and add the current contents of 'data' to it
        delete_data : bool
            if an AtomsData database exists, delete it if this variable is True (default is True)
        continue_train : bool
            if there are previous SchNet log/checkpoints, determines whether to use these to reinitialize training (default is False)
        """
        self.data = data
        self.energy = np.asarray(energy)
        self.energy = self.energy.reshape(self.energy.shape[0], 1)
        self.forces = forces
        if name is None:
            self.name = 'train'
        else:
            self.name = name

        self.use_current_db = use_current_db
        self.delete_data = delete_data
        self.continue_train = continue_train
        self.add_data = add_data

    def construct_train(self):
        raise NotImplementedError

    def train(self):
        """
        This applies to all trainers and can be inherited by all child classes
        """
        self.trainer.fit(self.learn_task, self.data_module)

class Train_NNIntra(Train_NN):
    """
    Class to set up and perform training for the intramolecular neural network
    """
    def __init__(self, 
            data, 
            energy, 
            forces, 
            openmm=None, 
            name=None, 
            use_current_db=False, 
            delete_data=True, 
            continue_train=False,
            add_data=False):

        super(Train_NNIntra, self).__init__(data, 
                energy, 
                forces, 
                name=name, 
                use_current_db=use_current_db, 
                delete_data=delete_data, 
                continue_train=continue_train,
                add_data=add_data)

        self.openmm = openmm

        #If saptff object is passed in, remove Morse potential energy/force
        if self.openmm is not None and not self.use_current_db:
            self.subtract_ff()

    def subtract_ff(self):
        """
        Remove force field energy and force contributions to total energy/force
        """

        #Energy expression only returns the energy contributions from OpenMM corresponding to this force
        ff_energy = []
        ff_forces = []
        for i, frame in enumerate(self.data):
            print(f"Structure {i+1}/{len(self.data)}")
            self.openmm.set_initial_positions(frame.get_positions())
            self.openmm.calculate(atoms=frame)
            omm_energy = self.openmm.get_potential_energy()
            omm_forces = self.openmm.get_forces()
            ff_energy.append(omm_energy)
            ff_forces.append(omm_forces)
        
        ff_energy = np.asarray(ff_energy)
        ff_energy = ff_energy.reshape(ff_energy.shape[0], 1)
        ff_forces = np.asarray(ff_forces)
        self.energy = np.asarray(self.energy)
        self.energy -= ff_energy

        if len(self.forces):
            self.forces -= ff_forces

    def construct_model(self,
            representation="painn",
            n_atom_basis=128,
            n_interactions=3,
            n_gaussians=30,
            cutoff=8.0,
            batch_size=100,
            num_train=0.8,
            num_val=0.15,
            num_test=0.05,
            lr=0.0005,
            mu=None,
            beta=None,
            reacting_atom_parent=None,
            reacting_atom_dissoc=None,
            keep_split=False,
            train_dir="lightning_logs",
            monitor_term="val_loss",
            num_gpu=1,
            num_epochs=4000,
            accelerator="cuda",
            ):
        """
        Parameters
        -----------
        representation : str
            NN representation used for Intra NN
        n_atom_basis : int
            vector length for representing atom embeddings
        n_interactions : int
            Number of interaction layers for PaiNN/SchNet
        n_gaussians : int
            Number of gaussians to use for expanding distances
        cutoff : float
            max distance for atoms to be considered neighbors
        batch_size : int
            Number of samples in dataset to pass in at once
        num_train : float
            Fraction of training set to use as training
        num_val : float
            Fraction of traning set to use as validation
        num_test : float
            Fraction of training set to use as test
        lr : float
            Learning rate
        mu : optional, float
            mu for Fermi-Dirac damping function
        beta : optional, float
            exponent for Fermi-Dirac damping function
        reacting_atom_parent : list
            atom index for Fermi-Dirac damping function
        reacting_atom_dissoc : list
            atom index for Fermi-Dirac damping function
        keep_split : bool
            Whether to keep current train-val-test split or build new one
        train_dir : str
            Where to store models and output
        monitor_term : str
            which term to monitor during training and change lr based off of
        num_gpu : int
            Number of gpus to train on
        num_epochs : int
            Max number of epochs
        accelerator : str
            Where to run training
        """

        if not self.use_current_db or self.add_data:
            #Get properties
            p_list = []
            for i, energy in enumerate(self.energy):
                prop = {'energy': energy}
                if len(self.forces):
                    prop['forces'] = self.forces[i]
                prop['training'] = np.asarray([1.0])
                p_list.append(prop)

        if not self.use_current_db and not self.add_data:
            if os.path.isfile(f'{self.name}.db'): os.remove(f'{self.name}.db')
            
            db = ASEAtomsData.create(f'{self.name}.db', 
                distance_unit='A', 
                property_unit_dict={'energy': 'kJ/mol', 'forces': 'kJ/mol/A', 'training': None})
            
            db.add_systems(p_list, self.data)
        
        if self.add_data:
            db = ASEAtomsData(f"{self.name}.db")
            db.add_systems(p_list, self.data)
        
        if not keep_split:
            if os.path.isfile('split.npz'): os.remove('split.npz')

        transforms = [TorchNeighborList(cutoff)]
        if mu:
            fd_setup = FDSetup_SchNet(reacting_atom_parent, reacting_atom_dissoc)

            transforms.append(fd_setup)
        
        transforms.append(CastTo32())

        self.data_module = AtomsDataModule(f'{self.name}.db', batch_size, num_train=num_train, num_val=num_val, num_test=num_test, transforms=transforms, split_file='split.npz', pin_memory=True)

        dist_expand = GaussianRBF(30, cutoff, 0.0, trainable=False)

        if representation == "PaiNN":
            if mu:
                rep = sch.representation.PaiNN_Damped(
                    n_atom_basis=n_atom_basis,
                    n_interactions=n_interactions,
                    radial_basis=dist_expand,
                    cutoff_fn=sch.nn.cutoff.CosineCutoff(cutoff),
                    damping_cutoff=mu,
                    beta=beta
                    )

                output = sch.atomistic.Atomwise_Damped(n_in=128)
                forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

            else:
                rep = sch.representation.PaiNN(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                radial_basis=dist_expand,
                cutoff_fn=sch.nn.cutoff.CosineCutoff(cutoff),
                )

                output = sch.atomistic.Atomwise(n_in=128)
                forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')
        
        elif representation == "SchNet":
            if mu:
                rep = sch.representation.SchNetDamped(
                    n_atom_basis=n_atom_basis,
                    n_interactions=n_interactions,
                    radial_basis=dist_expand,
                    cutoff_fn=sch.nn.cutoff.CosineCutoff(cutoff),
                    damping_cutoff=mu,
                    beta=beta,
                    activation=sch.nn.activations.shifted_softplus
                    )
                
                output = sch.atomistic.Atomwise_Damped(n_in=128, activation=sch.nn.activations.shifted_softplus)
                forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

            else:
                rep = sch.representation.SchNet(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                radial_basis=dist_expand,
                cutoff_fn=sch.nn.cutoff.CosineCutoff(cutoff),
                activation=sch.nn.activations.shifted_softplus,
                )

                output = sch.atomistic.Atomwise(n_in=128, activation=sch.nn.activations.shifted_softplus)
                forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')    
        else:
            print(f"Representation {representation} is not available, enter SchNet or PaiNN in the input file")
            sys.exit()

        mae_dict = {'MAE': torchmetrics.MeanAbsoluteError()}
        model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=0.1, metrics=mae_dict)
        model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=0.9, metrics=mae_dict)

        if mu:
            damping = sch.atomistic.response.Damping()
            output_modules = [output, damping, forces]
        else:
            output_modules = [output, forces]

        model = sch.model.NeuralNetworkPotential(
            rep,
            input_modules=[sch.atomistic.PairwiseDistances()],
            output_modules=output_modules)

        #old_model = torch.load("ch3cl_model")
        #old_params = {}
        #for i, param in enumerate(old_model.named_parameters()):
        #    old_params[i] = param[1]
        #for i, param in enumerate(model.named_parameters()):
        #    param[1].data = old_params[i]

        #torch.save(model, "ch3cl_model_40")
        
        outputs = [model_eng, model_force]
        optimizer_args = {'lr': lr}
        self.learn_task = sch.task.AtomisticTask(
            model,
            outputs=outputs,
            optimizer_cls=Adam,
            optimizer_args=optimizer_args,
            scheduler_cls=sch.train.ReduceLROnPlateau,
            scheduler_args={'min_lr': 1e-6},
            scheduler_monitor=monitor_term
        )

        self.trainer = Trainer(devices=num_gpu, max_epochs=num_epochs, callbacks=[ModelCheckpoint(f"{train_dir}/best_model", monitor=monitor_term)], accelerator=accelerator, default_root_dir=train_dir)

class Train_NNInter(Train_NN):
    """
    Class to set up and perform training for the intermolecular neural network
    """
    def __init__(self, data, energy, forces, openmm=None, name=None, use_current_db=False, delete_data=True, continue_train=False):
        super(Train_NNInter, self).__init__(data, energy, forces, name=name, use_current_db=use_current_db, delete_data=delete_data, continue_train=continue_train)

        self.openmm = openmm
        
        #If saptff object is passed in, remove Morse potential energy/force
        if self.openmm is not None and not self.use_current_db:
            self.subtract_ff()

    def subtract_ff(self):
        """
        Remove force field energy and force contributions to total energy/force
        """
        self.energy_expression = {'CustomBondForce': 'step(r-rhyper)*((r-rhyper)*khyper)^powh'}
        ff_energy = []
        ff_forces = []
        for i, frame in enumerate(self.data):
            print(f"Structure {i}/{len(self.data)}")
            self.openmm.set_initial_positions(frame.get_positions())
            self.openmm.set_xyz(frame.get_positions())
            energy, forces = self.openmm.compute_energy_component(['CustomBondForce', 'NonbondedForce', 'CustomNonbondedForce', 'DrudeForce'], energy_expression=self.energy_expression)
            ff_energy.append(energy)
            ff_forces.append(forces)

        ff_energy = np.asarray(ff_energy)
        ff_forces = np.asarray(ff_forces)
        ff_energy = np.expand_dims(ff_energy, -1)
        
        if len(self.energy):
            self.energy -= ff_energy
        if len(self.forces):
            self.forces -= ff_forces

    def construct_model(self,
            n_in=128,
            n_acsf=43,
            n_ap=21,
            elements=frozenset((1, 6, 7, 8)),
            batch_size=100,
            num_train=0.8,
            num_val=0.15,
            num_test=0.05,
            lr=0.0005,
            rho_tradeoff=0.9,
            reacting_atom_parent=None,
            reacting_atom_dissoc=None,
            mu=None,
            beta=30.0,
            keep_split=False,
            monitor_term="val_loss",
            num_gpu=1,
            num_epochs=4000,
            train_dir="lightning_logs_inter",
            accelerator='cuda'
            ):
        """
        Parameters
        -----------
        n_in : int
            vector size for dense layers
        n_acsf : int
            Number of radial symmetry functions
        n_ap : int
            Number of atom pair symmetry functions
        elements : set
            Set of elements included in the NN
        batch_size : int
            Number of samples in dataset to pass in at once
        num_train : float
            Fraction of training set to use as training
        num_val : float
            Fraction of traning set to use as validation
        num_test : float
            Fraction of training set to use as test
        lr : float
            Learning rate
        mu : optional, float
            mu for Fermi-Dirac damping function
        beta : optional, float
            exponent for Fermi-Dirac damping function
        reacting_atom_parent : list
            atom index for Fermi-Dirac damping function
        reacting_atom_dissoc : list
            atom index for Fermi-Dirac damping function
        keep_split : bool
            Whether to keep current train-val-test split or build new one
        train_dir : str
            Where to store models and output
        monitor_term : str
            which term to monitor during training and change lr based off of
        num_gpu : int
            Number of gpus to train on
        num_epochs : int
            Max number of epochs
        accelerator : str
            Where to run training
        """

        if not self.use_current_db:

            p_list = []
            for i, data in enumerate(self.data):
                prop = {}
                if len(self.energy):
                    prop['energy'] = self.energy[i]
                if len(self.forces):
                    prop['forces'] = self.forces[i]
                prop['training'] = np.asarray([1.0])
                p_list.append(prop)

        atomic_numbers_all = []
        for atom in self.openmm.pdb.topology.atoms():
            atomic_numbers_all.append(atom.element.atomic_number)
        atomic_numbers_all = np.asarray(atomic_numbers_all)
        elements = frozenset(atomic_numbers_all)

        if not self.use_current_db:
            if os.path.isfile(f'{self.name}.db'): os.remove(f'{self.name}.db')

        property_unit_dict = {}
        if len(self.energy):
            property_unit_dict['energy'] = 'kJ/mol'
        if len(self.forces):
            property_unit_dict['forces'] = 'kJ/mol/A'
        property_unit_dict['training'] = None
        
        if not self.use_current_db:
            db = ASEAtomsData.create(f'{self.name}.db', distance_unit='A', property_unit_dict=property_unit_dict)
            db.add_systems(p_list, self.data)

        if not keep_split:
            if os.path.isfile('split.npz'): os.remove('split.npz')

        syms = self.data[-1].get_atomic_numbers()
        res_dict = self.openmm.res_list()
        ZA = syms[res_dict[0]]
        ZB = syms[res_dict[1]]
        if self.data[-1].pbc.any():
            transforms = [APNetPBCNeighborList(ZA,ZB)]
        else:
            transforms = [APNetNeighborList(ZA, ZB)]
        
        if mu:
            fd_setup = FDSetup(reacting_atom_parent, reacting_atom_dissoc)

            transforms.append(fd_setup)

        transforms.append(CastTo32())

        self.data_module = AtomsDataModule(f'{self.name}.db', batch_size, num_train=num_train, num_val=num_val, num_test=num_test, transforms=transforms, split_file='split.npz', pin_memory=True)

        #Default settings should be fine here
        apnet = sch.representation.APNet(
            n_ap=n_ap,
            elements=elements,
            cutoff_radius=4.,
            cutoff_radius2=4.,
            sym_cut=4.5,
            cutoff=sch.nn.cutoff.CosineCutoff,
            mu=mu,
            beta=beta,
        )

        output = sch.atomistic.Pairwise(n_in=n_in, n_hidden=n_in, n_layers=4, elements=elements)
        forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

        mae_dict = {'MAE': torchmetrics.MeanAbsoluteError()}
        
        #Get proper loss functions based on properties in training set
        outputs = []
        if len(self.energy) and len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=0.1, metrics=mae_dict)
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=0.9, metrics=mae_dict)
            outputs = [model_eng, model_force]
        elif len(self.energy) and not len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_eng]
        elif len(self.forces):
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_force]
        else:
            raise Exception("Need to have energy or force properties")

        damping = sch.atomistic.response.Damping()
        model = sch.model.PairwiseModel(
            apnet,
            input_modules=[sch.atomistic.APNetFeatures()],
            output_modules=[output, damping, forces])

        optimizer_args = {'lr': lr}
        self.learn_task = sch.task.AtomisticTask(
            model,
            outputs=outputs,
            optimizer_cls=Adam,
            optimizer_args=optimizer_args,
            scheduler_cls=sch.train.ReduceLROnPlateau,
            scheduler_args={'min_lr': 1e-6},
            scheduler_monitor=monitor_term
        )
        
        self.trainer = Trainer(devices=num_gpu, max_epochs=num_epochs, callbacks=[ModelCheckpoint(f"{train_dir}/best_model", monitor=monitor_term)], accelerator=accelerator, default_root_dir=train_dir)

class Train_NNHij(Train_NN):
    """
    Train single coupling neural network
    """
    def __init__(self, data, energy, forces, diabats, name=None, use_current_db=False, continue_train=False):
        super(Train_NNHij, self).__init__(data, energy, forces, name=name, use_current_db=use_current_db, continue_train=continue_train)

        #List of Diabat objects
        self.diabats = diabats

        if not self.use_current_db:
            self._get_coupling_energy()

    def _get_coupling_energy(self):

        self.energy_expression = {}

        #Get list containing energy/force of H11 and H22
        h11_energy = []
        h22_energy = []
        h11_forces = []
        h22_forces = []
        for i, d in enumerate(self.data):
            print(i)
            energy, force = self.diabats[0].compute_energy_force(d)
            h11_energy.append(energy)
            h11_forces.append(force)
            
            energy, force = self.diabats[1].compute_energy_force(d)
            h22_energy.append(energy)
            h22_forces.append(force)

        h11_energy = np.asarray(h11_energy)
        h11_forces = np.asarray(h11_forces)
        h22_energy = np.asarray(h22_energy)
        h22_forces = np.asarray(h22_forces)

        #From H11 and H22 energy, get the off diagonal energy. Note that H11 and H22 cannot be lower in energy than the ground state reference energy. This may
        #happen due to the level of theory used for the diabats. Get the indices of dataset where this is the case
        h12_energy, subzero_diff_points = self._get_off_diag_energy(h11_energy, h22_energy)

        #Get H12 forces
        if self.forces is not None:
            h12_forces = self._get_off_diag_forces(h11_energy, h22_energy, h11_forces, h22_forces)

        if subzero_diff_points is not None:
            #For now, delete these from the dataset. Could also set the energy and force to 0.
            self.h12_energy = np.delete(h12_energy, subzero_diff_points[0], axis=0)
            if self.forces is not None:
                self.h12_forces = np.delete(h12_forces, subzero_diff_points[0], axis=0)
            self.data = [d for i, d in enumerate(self.data) if i not in subzero_diff_points[0]]
        else:
            self.h12_energy = h12_energy
            self.h12_forces = h12_forces

    def _get_off_diag_energy(self, h11_eng, h22_eng):
        """
        Parameters
        -----------
        h11_eng : np.ndarray
            contains H11 diabat energy
        h22_eng : np.ndarray
            contains H22 diabat energy

        Returns
        -----------
        h12 : np.ndarray
            contains H12 coupling energy, formed from reference energy and the diabat energies
        inds_subzero : np.ndarray
            locations in the data where H12 is nan
        """
        diff_h11 = h11_eng - self.energy
        diff_h22 = h22_eng - self.energy
        inds_subzero = np.where(np.logical_or(diff_h11 < 0, diff_h22 < 0))

        h12_sq = diff_h11 * diff_h22
        h12 = np.sqrt(h12_sq)
        return np.asarray(h12), inds_subzero

    def _get_off_diag_forces(self, h11_eng, h22_eng, h11_forces, h22_forces):
        """
        Parameters
        -----------
        h11_eng : np.ndarray
            contains H11 diabat energy
        h22_eng : np.ndarray
            contains H22 diabat energy
        h11_forces : np.ndarray
            contains H11 forces
        h22_forces : np.ndarray
            Contains H22 forces

        Returns
        -----------
        forces : np.ndarray
            contains H12 forces
        """
        #Form prefactor for the forces
        diff_h11 = h11_eng - self.energy
        diff_h22 = h22_eng - self.energy
        fac = 0.5 * (diff_h11 * diff_h22)**(-0.5) 

        #Ensure energy terms are the same shape as the force terms
        diff_h11_forces = h11_forces - self.forces
        diff_h22_forces = h22_forces - self.forces
        diff_h11 = np.repeat(diff_h11, diff_h11_forces.shape[1], axis=1)
        diff_h11 = np.expand_dims(diff_h11, -1)
        diff_h22 = np.repeat(diff_h22, diff_h22_forces.shape[1], axis=1)
        diff_h22 = np.expand_dims(diff_h22, -1)

        #product rule of the energy H12 expression
        dr = diff_h11 * diff_h22_forces + diff_h22 * diff_h11_forces
        fac = np.repeat(fac, dr.shape[1], axis=1)
        fac = np.expand_dims(fac, -1)

        #multiply the prefactor times dr to get the final force expression
        forces = fac * dr
        return forces

    def construct_model(self,
            n_in=128,
            n_acsf=43,
            n_ap=21,
            elements=frozenset((1, 6, 7, 8)),
            batch_size=100,
            num_train=0.8,
            num_val=0.15,
            num_test=0.05,
            lr=0.0005,
            reacting_atom_parent_react=None,
            reacting_atom_dissoc_react=None,
            reacting_atom_parent_prod=None,
            reacting_atom_dissoc_prod=None,
            mu_react=None,
            beta_react=30.0,
            mu_prod=None,
            beta_prod=30.0,
            keep_split=False,
            monitor_term="val_loss",
            num_gpu=1,
            num_epochs=4000,
            train_dir="lightning_logs_coupling",
            accelerator='cuda'
            ):
        """
        Parameters
        -----------
        n_in : int
            vector size for dense layers
        n_acsf : int
            Number of radial symmetry functions
        n_ap : int
            Number of atom pair symmetry functions
        elements : set
            Set of elements included in the NN
        batch_size : int
            Number of samples in dataset to pass in at once
        num_train : float
            Fraction of training set to use as training
        num_val : float
            Fraction of traning set to use as validation
        num_test : float
            Fraction of training set to use as test
        lr : float
            Learning rate
        reacting_atom_parent_react : list
            atom index for Fermi-Dirac damping function for Reactant
        reacting_atom_dissoc_react : list
            atom index for Fermi-Dirac damping function for Reactant
        reacting_atom_parent_prod : list
            atom index for Fermi-Dirac damping function for Product
        reacting_atom_dissoc_prod : list
            atom index for Fermi-Dirac damping function for Product
        mu_react : optional, float
            mu for Fermi-Dirac damping function for Reactant
        beta_react : float
            beta for Fermi-Dirac damping function for Reactant
        mu_prod : optional, float
            mu for Fermi-Dirac damping function for Product
        beta_prod : float
            beta for Fermi-Dirac damping function for Product
        keep_split : bool
            Whether to keep current train-val-test split or build new one
        train_dir : str
            Where to store models and output
        monitor_term : str
            which term to monitor during training and change lr based off of
        num_gpu : int
            Number of gpus to train on
        num_epochs : int
            Max number of epochs
        accelerator : str
            Where to run training
        """

        if not self.use_current_db:

            p_list = []
            for i, data in enumerate(self.data):
                prop = {}
                if len(self.energy):
                    prop['energy'] = self.h12_energy[i]
                if len(self.forces):
                    prop['forces'] = self.h12_forces[i]
                prop['training'] = np.asarray([1.0])
                p_list.append(prop)

        atomic_numbers_all = []
        for atom in self.diabats[0].openmm.pdb.topology.atoms():
            atomic_numbers_all.append(atom.element.atomic_number)
        atomic_numbers_all = np.asarray(atomic_numbers_all)
        elements = frozenset(atomic_numbers_all)

        if not self.use_current_db:
            if os.path.isfile(f'{self.name}.db'): os.remove(f'{self.name}.db')

        property_unit_dict = {}
        if len(self.energy):
            property_unit_dict['energy'] = 'kJ/mol'
        if len(self.forces):
            property_unit_dict['forces'] = 'kJ/mol/A'
        property_unit_dict['training'] = None

        if not self.use_current_db:
            db = ASEAtomsData.create(f'{self.name}.db', distance_unit='A', property_unit_dict=property_unit_dict)
            db.add_systems(p_list, self.data)

        if not keep_split:
            if os.path.isfile('split.npz'): os.remove('split.npz')
        
        transforms = [APOffDiagNeighborList()]
        if mu_react:
            fd_setup = FDSetup_OffDiag(reacting_atom_parent_react, reacting_atom_dissoc_react, reacting_atom_parent_prod, reacting_atom_dissoc_prod)

            transforms.append(fd_setup)

        transforms.append(CastTo32())

        self.data_module = AtomsDataModule(f'{self.name}.db', batch_size, num_train=num_train, num_val=num_val, num_test=num_test, transforms=transforms, split_file='split.npz', pin_memory=True)

        #Default settings should again be good enough here
        apnet = sch.representation.APNet_OffDiag(
            n_ap=n_ap,
            elements=elements,
            cutoff_radius=8.,
            cutoff_radius2=4.,
            sym_cut=5.5,
            cutoff=sch.nn.cutoff.CosineCutoff,
            mu_react=mu_react,
            beta_react=beta_react,
        )

        output = sch.atomistic.Pairwise_OffDiag(n_in=n_in, n_hidden=n_in, n_layers=4, elements=elements)
        forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

        mae_dict = {'MAE': torchmetrics.MeanAbsoluteError()}

        outputs = []
        if len(self.energy) and len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=0.1, metrics=mae_dict)
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=0.9, metrics=mae_dict)
            outputs = [model_eng, model_force]
        elif len(self.energy) and not len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_eng]
        elif len(self.forces):
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_force]
        else:
            raise Exception("Need to have energy or force properties")

        damping = sch.atomistic.response.Damping()
        model = sch.model.PairwiseModel_OffDiag(
            apnet,
            input_modules=[sch.atomistic.APNetFeatures_OffDiag()],
            output_modules=[output, damping, forces])

        optimizer_args = {'lr': lr}
        self.learn_task = sch.task.AtomisticTask(
            model,
            outputs=outputs,
            optimizer_cls=Adam,
            optimizer_args=optimizer_args,
            scheduler_cls=sch.train.ReduceLROnPlateau,
            scheduler_args={'min_lr': 1e-6},
            scheduler_monitor=monitor_term
        )

        self.trainer = Trainer(devices=num_gpu, max_epochs=num_epochs, callbacks=[ModelCheckpoint(f"{train_dir}/best_model", monitor=monitor_term)], accelerator=accelerator, default_root_dir=train_dir)

class Train_NNQMMM(Train_NN):
    """
    Train H11 inter NN, H22 inter NN and Hij NN simultaneously
    """
    def __init__(self, 
            data, 
            energy, 
            forces, 
            diabats, 
            e_field=[], 
            e_potential=[], 
            name=None, 
            use_current_db=False, 
            continue_train=False,
            add_data=False):
        super(Train_NNQMMM, self).__init__(data, energy, forces, name=name, use_current_db=use_current_db, continue_train=continue_train, add_data=add_data)
        
        #List of Diabat objects
        self.diabats = diabats
        self.e_field = e_field
        self.e_potential = e_potential
        self.nn_inter_models = []
        self.nn_inter_neighborlist = []
        self.nn_hij_neighborlist = []
        self.nn_inter_fermi_dirac = []
        self.nn_hij_fermi_dirac = []
        if not self.use_current_db:
            self._get_diabats_fixed()

    def _get_diabats_fixed(self):
        """
        Get OpenMM and NNIntra energies for each diabat
        """

        #Get list containing energy/force of H11 and H22
        h11_energy = []
        h22_energy = []
        h11_forces = []
        h22_forces = []
        potential_solvent_A_h11 = []
        potential_solvent_B_h11 = []
        potential_solvent_A_h22 = []
        potential_solvent_B_h22 = []
        potential_solvent = []
        reorder_indices = []
        reverse_reorder_indices = []

        for i, d in enumerate(self.data):
            print(f"Structure {i+1}/{len(self.data)}")

            total_forces = np.zeros_like(d.get_positions())
            nnintra_h11, nnintra_forces_h11 = self.diabats[0].compute_nn_intra(d, total_forces)
            openmm_h11, openmm_forces_h11 = self.diabats[0].compute_openmm_energy(d)

            total_forces = np.zeros_like(d.get_positions())
            new_atoms, reorder_list = self.diabats[1].reorder_graph.reorder(d)
            nnintra_h22, nnintra_forces_h22 = self.diabats[1].compute_nn_intra(new_atoms, total_forces)
            openmm_h22, openmm_forces_h22 = self.diabats[1].compute_openmm_energy(new_atoms)

            potential_h11 = 0
            potential_h22 = 0
            res_list = self.diabats[0].openmm.res_list()
            monomer_A_h11 = res_list[0]
            monomer_B_h11 = res_list[1]

            #Monomer A and B electric potentials
            potential_A_h11 = np.zeros((total_forces[monomer_A_h11].shape[0]))
            potential_B_h11 = np.zeros((total_forces[monomer_B_h11].shape[0]))

            res_list = self.diabats[1].openmm.res_list()
            monomer_A_h22 = res_list[0]
            monomer_B_h22 = res_list[1]

            potential_A_h22 = np.zeros((total_forces[monomer_A_h22].shape[0]))
            potential_B_h22 = np.zeros((total_forces[monomer_B_h22].shape[0]))
            potential_solv = np.zeros((total_forces.shape[0]))
            #Add electrical potential energy for both diabats
            if len(self.e_potential[i]):
                potential = self.e_potential[i]
                potential = np.asarray(potential)
                potential = potential[None]
                charges = self.diabats[0].openmm.get_charges()
                for atom_potential in potential:
                    potential_A_h11 += atom_potential[monomer_A_h11]
                    potential_B_h11 += atom_potential[monomer_B_h11]
                    potential_solv += atom_potential
                    for p, charge in zip(atom_potential, charges):
                        potential_h11 += p*charge
                
                charges = self.diabats[1].openmm.get_charges()
                for atom_potential in potential:
                    reorder_potential = atom_potential[reorder_list]
                    potential_A_h22 += reorder_potential[monomer_A_h22]
                    potential_B_h22 += reorder_potential[monomer_B_h22]
                    for p, charge in zip(reorder_potential, charges):
                        potential_h22 += p*charge

            field_h11 = np.zeros_like(openmm_forces_h11)
            field_h22 = np.zeros_like(openmm_forces_h22)
            
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
                    reorder_field = atom_field[reorder_list]
                    for j, (f, charge) in enumerate(zip(reorder_field, charges)):
                        field_h22[j] -= f*charge
            
            energy = nnintra_h11 + openmm_h11 + potential_h11 + self.diabats[0].shift
            forces = nnintra_forces_h11 + openmm_forces_h11 + field_h11
            h11_energy.append(energy)
            h11_forces.append(forces)

            energy = nnintra_h22 + openmm_h22 + potential_h22 + self.diabats[1].shift
            forces = nnintra_forces_h22 + openmm_forces_h22 + field_h22
            forces = reorder(forces, reorder_list)
            h22_energy.append(energy)
            h22_forces.append(forces)

            potential_solvent_A_h11.append(potential_A_h11)
            potential_solvent_B_h11.append(potential_B_h11)
            potential_solvent_A_h22.append(potential_A_h22)
            potential_solvent_B_h22.append(potential_B_h22)
            potential_solvent.append(potential_solv)

            reorder_indices.append(reorder_list)
            reverse_index = [reorder_list.index(i) for i in range(len(reorder_list))]
            reverse_reorder_indices.append(reverse_index)

        self.h11_energy = np.asarray(h11_energy)
        self.h11_forces = np.asarray(h11_forces)
        self.h22_energy = np.asarray(h22_energy)
        self.h22_forces = np.asarray(h22_forces)
        self.reorder_indices = np.asarray(reorder_indices)
        self.reverse_reorder_indices = np.asarray(reverse_reorder_indices)
        self.potential_solvent_A_h11 = np.asarray(potential_solvent_A_h11)
        self.potential_solvent_B_h11 = np.asarray(potential_solvent_B_h11)
        self.potential_solvent_A_h22 = np.asarray(potential_solvent_A_h22)
        self.potential_solvent_B_h22 = np.asarray(potential_solvent_B_h22)
        self.potential_solvent = np.asarray(potential_solvent)

    def construct_inter_model( 
            self,
            n_in=128,
            n_acsf=43,
            n_ap=21,
            elements=frozenset((1, 6, 7, 8)),
            reacting_atom_parent=None,
            reacting_atom_dissoc=None,
            mu=None,
            beta=30.0,
            diabat=None
            ):
        """
        Build NN Inter models for training

        Parameters
        -----------
        n_in : int
            vector size for dense layers
        n_acsf : int
            Number of radial symmetry functions
        n_ap : int
            Number of atom pair symmetry functions
        elements : set
            Set of elements included in the NN
        mu : optional, float
            mu for Fermi-Dirac damping function
        beta : optional, float
            exponent for Fermi-Dirac damping function
        reacting_atom_parent : list
            atom index for Fermi-Dirac damping function
        reacting_atom_dissoc : list
            atom index for Fermi-Dirac damping function
        diabat : None or int
            Index of the diabat the NN will be used for
        """

        res_list = self.diabats[diabat].openmm.res_list()

        atomic_numbers = []
        for res in self.diabats[diabat].openmm.pdb.topology.residues():
            numbers = []
            for atom in res.atoms():
                numbers.append(atom.element.atomic_number)
            atomic_numbers.append(numbers)

        atomic_numbers_all = []
        for atom in self.diabats[diabat].openmm.pdb.topology.atoms():
            atomic_numbers_all.append(atom.element.atomic_number)
        atomic_numbers_all = np.asarray(atomic_numbers_all)
        elements = frozenset(atomic_numbers_all)
        
        ZA = np.asarray(atomic_numbers[0])
        ZB = np.asarray(atomic_numbers[1])
        if self.data[-1].pbc.any():
            self.nn_inter_neighborlist.append(APNetPBCNeighborList(ZA,ZB))
        else:
            self.nn_inter_neighborlist.append(APNetNeighborList(ZA, ZB))

        if mu:
            fd_setup = FDSetup(reacting_atom_parent, reacting_atom_dissoc)

            self.nn_inter_fermi_dirac.append(fd_setup)

        apnet = sch.representation.APNet(
            n_ap=n_ap,
            elements=elements,
            cutoff_radius=4.,
            cutoff_radius2=4.,
            sym_cut=4.5,
            cutoff=sch.nn.cutoff.CosineCutoff,
            mu=mu,
            beta=beta,
        )
        
        output = sch.atomistic.PairwiseSolvent(n_in=n_in, n_hidden=n_in, n_layers=4, elements=elements)
        forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

        damping = sch.atomistic.response.Damping()
        model = sch.model.PairwiseModelSolvent(
            apnet,
            input_modules=[sch.atomistic.APNetFeatures()],
            output_modules=[output, damping, forces])

        self.nn_inter_models.append(model)

    def construct_hij_model(self,
            n_in=128,
            n_acsf=43,
            n_ap=21,
            elements=frozenset((1, 6, 7, 8)),
            reacting_atom_parent_react=None,
            reacting_atom_dissoc_react=None,
            reacting_atom_parent_prod=None,
            reacting_atom_dissoc_prod=None,
            mu_react=None,
            beta_react=30.0,
            mu_prod=None,
            beta_prod=30.0,
            ):
        """
        Parameters
        -----------
        n_in : int
            vector size for dense layers
        n_acsf : int
            Number of radial symmetry functions
        n_ap : int
            Number of atom pair symmetry functions
        elements : set
            Set of elements included in the NN
        reacting_atom_parent_react : list
            atom index for Fermi-Dirac damping function for Reactant
        reacting_atom_dissoc_react : list
            atom index for Fermi-Dirac damping function for Reactant
        reacting_atom_parent_prod : list
            atom index for Fermi-Dirac damping function for Product
        reacting_atom_dissoc_prod : list
            atom index for Fermi-Dirac damping function for Product
        mu_react : optional, float
            mu for Fermi-Dirac damping function for Reactant
        beta_react : float
            beta for Fermi-Dirac damping function for Reactant
        mu_prod : optional, float
            mu for Fermi-Dirac damping function for Product
        beta_prod : float
            beta for Fermi-Dirac damping function for Product
        """
        if self.data[-1].pbc.any():
            self.nn_hij_neighborlist.append(APOffDiagPBCNeighborList())
        else:
            self.nn_hij_neighborlist.append(APOffDiagNeighborList())

        elements = self.data[-1].get_atomic_numbers()
        elements = frozenset(elements)

        if mu_react:
            fd_setup = FDSetup_OffDiag(reacting_atom_parent_react, reacting_atom_dissoc_react, reacting_atom_parent_prod, reacting_atom_dissoc_prod)

            self.nn_hij_fermi_dirac.append(fd_setup)

        #Default settings here should be fine
        apnet = sch.representation.APNet_OffDiag(
            n_ap=n_ap,
            elements=elements,
            cutoff_radius=8.,
            cutoff_radius2=4.,
            sym_cut=5.5,
            cutoff=sch.nn.cutoff.CosineCutoff,
            mu_react=mu_react,
            beta_react=beta_react,
            mu_prod=mu_prod,
            beta_prod=beta_prod
        )

        output = sch.atomistic.Pairwise_OffDiagSolvent(n_in=n_in, n_hidden=n_in, n_layers=4, elements=elements)
        forces = sch.atomistic.response.Forces(calc_forces=True, energy_key='y', force_key='dr_y')

        damping = sch.atomistic.response.Damping()
        self.hij_model = sch.model.PairwiseModel_OffDiagSolvent(
            apnet,
            input_modules=[sch.atomistic.APNetFeatures_OffDiag()],
            output_modules=[output, damping, forces]
            )

    def construct_database(self,
            batch_size=75,
            num_train=0.8,
            num_val=0.15,
            num_test=0.05,
            lr=0.0005,
            keep_split=False,
            monitor_term="val_loss",
            num_gpu=1,
            num_epochs=4000,
            train_dir="lightning_logs_coupling",
            accelerator='cuda'
            ):
        """
        Parameters
        -----------
        batch_size : int
            Number of samples in dataset to pass in at once
        num_train : float
            Fraction of training set to use as training
        num_val : float
            Fraction of traning set to use as validation
        num_test : float
            Fraction of training set to use as test
        lr : float
            Learning rate
        keep_split : bool
            Whether to keep current train-val-test split or build new one
        train_dir : str
            Where to store models and output
        monitor_term : str
            which term to monitor during training and change lr based off of
        num_gpu : int
            Number of gpus to train on
        num_epochs : int
            Max number of epochs
        accelerator : str
            Where to run training
        """
        #Assemble various properties for DB
        if not self.use_current_db:
            p_list = []
            for i, data in enumerate(self.data):
                prop = {}
                if len(self.energy):
                    prop['energy'] = self.energy[i]
                if len(self.forces):
                    prop['forces'] = self.forces[i]
                prop['h11_energy'] = self.h11_energy[i]
                prop['h22_energy'] = self.h22_energy[i]
                prop['h11_forces'] = self.h11_forces[i]
                prop['h22_forces'] = self.h22_forces[i]
                prop['reorder_indices'] = self.reorder_indices[i]
                prop['reverse_reorder_indices'] = self.reverse_reorder_indices[i]
                prop['potential_solvent_A'] = self.potential_solvent_A_h11[i]
                prop['potential_solvent_B'] = self.potential_solvent_B_h11[i]
                prop['potential_solvent_A_2'] = self.potential_solvent_A_h22[i]
                prop['potential_solvent_B_2'] = self.potential_solvent_B_h22[i]
                prop['potential_solvent_ij'] = self.potential_solvent[i]
                p_list.append(prop)
            if os.path.isfile(f'{self.name}.db'): os.remove(f'{self.name}.db')
        
        #Units don't really mater, they just have to all be present in the dictionary
        property_unit_dict = {}
        if len(self.energy):
            property_unit_dict['energy'] = 'kJ/mol'
        if len(self.forces):
            property_unit_dict['forces'] = 'kJ/mol/A'
        property_unit_dict['h11_energy'] = 'kJ/mol'
        property_unit_dict['h22_energy'] = 'kJ/mol'
        property_unit_dict['potential_solvent_A'] = 'kJ/mol'
        property_unit_dict['potential_solvent_B'] = 'kJ/mol'
        property_unit_dict['potential_solvent_A_2'] = 'kJ/mol'
        property_unit_dict['potential_solvent_B_2'] = 'kJ/mol'
        property_unit_dict['potential_solvent_ij'] = 'kJ/mol'
        property_unit_dict['h11_forces'] = 'kJ/mol/A'
        property_unit_dict['h22_forces'] = 'kJ/mol/A'
        property_unit_dict['reorder_indices'] = None
        property_unit_dict['reverse_reorder_indices'] = None

        if not self.use_current_db and not self.add_data:
            if os.path.isfile(f'{self.name}.db'): os.remove(f'{self.name}.db')
            db = ASEAtomsData.create(f'{self.name}.db',
                distance_unit='A',
                property_unit_dict=property_unit_dict)

            db.add_systems(p_list, self.data)

        if self.add_data:
            db = ASEAtomsData(f"{self.name}.db")
            db.add_systems(p_list, self.data)

        if not keep_split:
            if os.path.isfile('split.npz'): os.remove('split.npz')

        transforms = []
        neighbor_list = APSimultaneousNeighborList(self.nn_inter_neighborlist[0], self.nn_inter_neighborlist[1], self.nn_hij_neighborlist[0])
        transforms.append(neighbor_list)
        fermi_dirac = FD_Simultaneous(self.nn_inter_fermi_dirac[0], self.nn_inter_fermi_dirac[1], self.nn_hij_fermi_dirac[0])
        transforms.append(fermi_dirac)
        transforms.append(CastTo32())

        self.data_module = AtomsDataModule(f'{self.name}.db', batch_size, num_train=num_train, num_val=num_val, num_test=num_test, transforms=transforms, split_file='split.npz', pin_memory=True)

        mae_dict = {'MAE': torchmetrics.MeanAbsoluteError()}

        outputs = []
        if len(self.energy) and len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=0.1, metrics=mae_dict)
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=0.9, metrics=mae_dict)
            outputs = [model_eng, model_force]
        elif len(self.energy) and not len(self.forces):
            model_eng = sch.task.ModelOutput('y', target_property='energy', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_eng]
        elif len(self.forces):
            model_force = sch.task.ModelOutput('dr_y', target_property='forces', loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics=mae_dict)
            outputs = [model_force]
        else:
            raise Exception("Need to have energy or force properties")

        optimizer_args = {'lr': lr}
        self.learn_task = CustomTask(
            self.nn_inter_models[0],
            self.nn_inter_models[1],
            self.hij_model,
            outputs=outputs,
            optimizer_cls=Adam,
            optimizer_args=optimizer_args,
            scheduler_cls=sch.train.ReduceLROnPlateau,
            scheduler_args={'min_lr': 1e-6},
            scheduler_monitor=monitor_term
        )

        self.trainer = Trainer(devices=num_gpu, max_epochs=num_epochs, callbacks=[ModelCheckpoint(f"{train_dir}/best_model", monitor=monitor_term)], accelerator=accelerator, default_root_dir=train_dir)

