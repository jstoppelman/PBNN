#!/usr/bin/env python
from sklearn.metrics import mean_absolute_error
import numpy as np

ref_energy = np.load("ref_energy_nnintra.npy")
test_energy = np.load("test_energy_nnintra.npy")
mae = mean_absolute_error(ref_energy, test_energy)

print("Energy MAE = ", mae, " kJ/mol")

ref_forces = np.load("ref_forces_nnintra.npy").flatten()
test_forces = np.load("test_forces_nnintra.npy").flatten()
mae = mean_absolute_error(ref_forces, test_forces)

print("Forces MAE = ", mae, " kJ/mol/A")

