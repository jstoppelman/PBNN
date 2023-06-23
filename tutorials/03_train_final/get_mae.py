#!/usr/bin/env python
from sklearn.metrics import mean_absolute_error
import numpy as np

ref_energy = np.load("ref_pbnn_energy.npy")
test_energy = np.load("test_pbnn_energy.npy")
mae = mean_absolute_error(ref_energy, test_energy)

print("Energy MAE = ", mae, " kJ/mol")

ref_forces = np.load("ref_pbnn_forces.npy").flatten()
test_forces = np.load("test_pbnn_forces.npy").flatten()
mae = mean_absolute_error(ref_forces, test_forces)

print("Forces MAE = ", mae, " kJ/mol/A")

