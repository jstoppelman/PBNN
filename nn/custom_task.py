import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel
from schnetpack import properties
from schnetpack.task import ModelOutput, UnsupervisedModelOutput

__all__ = ["CustomTask"]

class CustomTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.
    We modified this class here in order to train multiple neural networks at the same time
    """

    def __init__(
        self,
        nninter_d1_model: AtomisticModel,
        nninter_d2_model: AtomisticModel,
        hij_model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            nninter_d1_model: the H11 nninter model
            nninter_d2_model: the H22 nninter model
            hij_model: the Hij model used for the coupling
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.nninter_d1_model = nninter_d1_model
        self.nninter_d2_model = nninter_d2_model
        self.hij_model = hij_model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        #Check if forces are required in any of the trained NNs
        self.grad_enabled = len(self.nninter_d1_model.required_derivatives) > 0 or len(self.nninter_d2_model.required_derivatives) > 0 or len(self.hij_model.required_derivatives) > 0 
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        """
        Attach data to each model
        """
        if stage == "fit":
            #Initialize transforms for all three models
            self.nninter_d1_model.initialize_transforms(self.trainer.datamodule)
            self.nninter_d2_model.initialize_transforms(self.trainer.datamodule)
            self.hij_model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Parameters
        -----------
        inputs : dict
            Contains the inputs for all three NNs. We extract them for the D2 inter NN
            and Hij NN

        Returns
        ----------
        results : dict
            Result dictionary containing energy and forces
        """
        #For the diabat 1 NN, we can leave the dataset ordered as it currently is
        #The input names can be those standardly used for the Inter NN model
        results_d1 = self.nninter_d1_model(inputs)
        
        #For D2, the inputs need to be renamed since they were stored in the diabat 2 ordering
        inputs_d2 = self.get_inputs_d2(inputs)
        results_d2 = self.nninter_d2_model(inputs_d2)

        #Similar for Hij
        inputs_hij = self.get_inputs_hij(inputs)
        results_hij = self.hij_model(inputs_hij)

        #Get total EVB/PBNN energy
        energy = self.calculate_energy(inputs, results_d1, results_d2, results_hij)
        #Get total EVB/PBNN forces
        forces = self.calculate_forces(inputs, results_d1, results_d2, results_hij)
        #Form results dictionary
        results = {'y': energy, 'dr_y': forces}

        return results

    def calculate_energy(self, inputs, results_d1, results_d2, results_hij):
        """
        Calculate the EVB energy from the determinant

        Parameters
        -----------
        inputs : dict
            Dictionary containing the PBNN energies and forces for the components not being trained
        results_d1 : dict
            Dictionary containing the H11 inter NN energy and forces
        results_d2 : dict
            Dictionary containing the H22 inter NN energy and forces
        results_hij : dict
            Dictionary containing the Hij NN energy and forces

        Returns
        ----------
        result : float
            Energy from the PBNN matrix
        """
        h11_ref_energy = inputs["h11_energy"]
        h22_ref_energy = inputs["h22_energy"]
        h11_nn_energy = results_d1["y"]
        h22_nn_energy = results_d2["y"]
        h12_energy = results_hij["y"]
        h11_energy = h11_ref_energy + h11_nn_energy
        h22_energy = h22_ref_energy + h22_nn_energy

        result = 0.5 * ((h11_energy + h22_energy) - torch.sqrt((h11_energy - h22_energy)**2 + 4 * h12_energy**2))
        return result

    def calculate_forces(self, inputs, results_d1, results_d2, results_hij):
        """
        Calculate the PBNN forces from the determinant (not Hellmann-Feynman)

        Parameters
        -----------
        inputs : dict
            Dictionary containing the PBNN energies and forces for the components not being trained
        results_d1 : dict
            Dictionary containing the H11 inter NN energy and forces
        results_d2 : dict
            Dictionary containing the H22 inter NN energy and forces
        results_hij : dict
            Dictionary containing the Hij NN energy and forces
        """
        n_atoms = inputs[properties.n_atoms]

        #Hii energies not being currently trained
        h11_ref_energy = inputs["h11_energy"]
        h22_ref_energy = inputs["h22_energy"]
        #Hii NN energies
        h11_nn_energy = results_d1["y"]
        h22_nn_energy = results_d2["y"]
        #Hij NN energies
        h12_energy = results_hij["y"]

        #Hii forces not currently being trained
        h11_ref_forces = inputs["h11_forces"]
        h22_ref_forces = inputs["h22_forces"]
        #Hii NN forces
        h11_nn_forces = results_d1["dr_y"]
        h22_nn_forces = results_d2["dr_y"]

        #Index to reorder H22 forces since they need to be in H11 order
        reverse_indices = inputs["reverse_reorder_indices"].long()
        h22_nn_forces = h22_nn_forces[reverse_indices]

        #Hij forces
        h12_forces = results_hij["dr_y"]

        #Total Hii energy and forces
        h11_energy = h11_ref_energy + h11_nn_energy
        h11_forces = h11_ref_forces + h11_nn_forces
        h22_energy = h22_ref_energy + h22_nn_energy
        h22_forces = h22_ref_forces + h22_nn_forces

        #Sum of both diagonal force components
        h11_h22_force = h11_forces + h22_forces

        #Reshape energies so they can be broadcasted to the forces
        h11_energy = h11_energy.unsqueeze(-1)
        h11_energy = torch.repeat_interleave(h11_energy, n_atoms[0], dim=0)
        h22_energy = h22_energy.unsqueeze(-1)
        h22_energy = torch.repeat_interleave(h22_energy, n_atoms[0], dim=0)
        h12_energy = h12_energy.unsqueeze(-1)
        h12_energy = torch.repeat_interleave(h12_energy, n_atoms[0], dim=0)
        
        #Numerator from the square root part of the force chain rule
        num = (h11_energy - h22_energy) * (h11_forces - h22_forces) + 4 * h12_energy * h12_forces
        #Final force result
        result = 0.5 * (h11_h22_force - num/torch.sqrt((h11_energy - h22_energy)**2 + 4 * h12_energy**2))
        return result
    
    def get_inputs_d2(self, inputs: Dict[str, torch.Tensor]):
        """
        Assemble a dictionary containing only inputs for the H22 NN

        Parameters
        -----------
        inputs : dict
            Dictionary containing all inputs

        Returns
        -----------
        tmp_dict : dict
            Dictionary containing inputs only pertaining to the D2 NN
        """
        tmp_inputs = {}

        reorder_indices = inputs["reorder_indices"].long()
        tmp_inputs[properties.Z] = inputs[properties.Z][reorder_indices]
        tmp_inputs[properties.cell] = inputs[properties.cell]
        tmp_inputs[properties.R] = inputs[properties.R][reorder_indices]
        tmp_inputs[properties.pbc] = inputs[properties.pbc]
        tmp_inputs[properties.n_atoms] = inputs[properties.n_atoms]
        tmp_inputs[properties.idx_m_pairs] = inputs["idx_m_2"]

        #Inputs corresponding to the D2 NN have _2 listed after their names
        for key in inputs.keys():
            if "_2" in key:
                og_name = key.split('_2')[0]
                tmp_inputs[og_name] = inputs[key]
        return tmp_inputs

    def get_inputs_hij(self, inputs: Dict[str, torch.Tensor]):
        """
        Assemble a dictionary containing only inputs for the Hij NN

        Parameters
        -----------
        inputs : dict
            Dictionary containing all inputs

        Returns
        -----------
        tmp_inputs : dict
            Dictionary containing inputs only pertaining to the Hij NN
        """
        tmp_inputs = {}

        tmp_inputs[properties.Z] = inputs[properties.Z]
        tmp_inputs[properties.cell] = inputs[properties.cell]
        tmp_inputs[properties.R] = inputs[properties.R]
        tmp_inputs[properties.pbc] = inputs[properties.pbc]
        tmp_inputs[properties.n_atoms] = inputs[properties.n_atoms]
        tmp_inputs[properties.idx_m_pairs] = inputs["idx_m_ij"]

        for key in inputs.keys():
            if "_ij" in key:
                og_name = key.split('_ij')[0]
                tmp_inputs[og_name] = inputs[key]
        return tmp_inputs

    def loss_fn(self, pred, batch):
        """
        Computes loss function 

        Parameters
        -----------
        pred : dict
            contains predicted values
        batch : dict
            contains the reference values

        Returns
        -----------
        Loss : float
            Current computed loss function
        """

        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        """
        Log current metrics 

        Parameters
        -----------
        pred : dict
            predicted values
        targets : dict
            reference values
        subset : str
            Train, test or validation subset
        """
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        """
        Not currently used for our purposes, but applies some constraint to the prediction wrt target

        Parameters
        -----------
        pred : dict
            Dictionary with predicted values
        targets : dict
            Dictionary with reference values

        Returns
        -----------
        pred : dict
            Dictionary with modified predicted values
        targets : dict
            Dictionary with modified target values
        """
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):
        """
        Computes predictions for training data

        Parameters
        -----------
        batch : dict
            Dictionary with batch input data
        batch_idx : int
            Used by PyTorch lightning to denote batch iteration

        Returns
        -----------
        loss : float
            Loss on training step
        """
        
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)
        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes predictions for validation data

        Parameters
        -----------
        batch : dict
            Dictionary with batch input data
        batch_idx : int
            Used by PyTorch lightning to denote batch iteration

        Returns
        -----------
        loss : float
            Loss on validation step
        """

        torch.set_grad_enabled(self.grad_enabled)
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """
        Computes predictions for test data

        Parameters
        -----------
        batch : dict
            Dictionary with batch input data
        batch_idx : int
            Used by PyTorch lightning to denote batch iteration

        Returns
        -----------
        loss : float
            Loss on test step
        """

        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        """
        Make predictions without doing a postprocessing step 
        (postprocessing refers to specific data transforms

        Parameters
        -----------
        batch : dict
            Batch containing the current data

        """
        pp1 = self.nninter_d1_model.do_postprocessing
        self.nninter_d1_model.do_postprocessing = False
        pp2 = self.nninter_d2_model.do_postprocessing
        self.nninter_d2_model.do_postprocessing = False
        pphij = self.hij_model.do_postprocessing 
        self.hij_model.do_postprocessing = False
        pred = self(batch)
        self.nninter_d1_model.do_postprocessing = pp1 
        self.nninter_d2_model.do_postprocessing = pp2
        self.hij_model.do_postprocessing = pphij
        return pred

    def configure_optimizers(self):
        """
        optimizer setup
        """
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ):
        """
        Optimize parameters

        Parameters
        -----------
        epoch : int
            Number of passes through the dataset that have been performed
        batch_idx : int
            Current Number of loops through the dataset so far
        optimizer : class
            PyTorch optimizer class
        optimizer_idx : int
            number of optimizers used so far
        optimizer_closure : function
            function only applicable to optimizers like LBFGS and Conjugate Gradient (see PyTorch documentation)
        on_tpu : bool
            Whether this calculation is run on TPU
        using_native_amp : bool
            Whether using pyTorch native precision handling package
        using_lbfgs : bool
            Whether using LBGFS optimizer
        """
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        """
        Save_model

        Parameters
        -----------
        path : str
            directory to save model
        do_postprocessing : bool
            Apply transforms to the computed results or not
        """
        prefix = path.split('/')[0]
        if self.trainer is None or self.trainer.strategy.local_rank == 0:
            pp_status = self.nninter_d1_model.do_postprocessing
            if do_postprocessing is not None:
                self.nninter_d1_model.do_postprocessing = do_postprocessing

            torch.save(self.nninter_d1_model, prefix+'/'+'best_model_inter_d1')

            self.nninter_d1_model.do_postprocessing = pp_status

            pp_status = self.nninter_d2_model.do_postprocessing
            if do_postprocessing is not None:
                self.nninter_d2_model.do_postprocessing = do_postprocessing

            torch.save(self.nninter_d2_model, prefix+'/'+'best_model_inter_d2')

            self.nninter_d2_model.do_postprocessing = pp_status

            pp_status = self.hij_model.do_postprocessing
            if do_postprocessing is not None:
                self.hij_model.do_postprocessing = do_postprocessing

            torch.save(self.hij_model, prefix+'/'+'best_model_hij')

            self.hij_model.do_postprocessing = pp_status


