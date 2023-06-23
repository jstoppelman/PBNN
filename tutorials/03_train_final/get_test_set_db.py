#!/usr/bin/env python
import numpy as np
import sys
from ase.db import connect
from ase.io import write

split = np.load("split.npz")["test_idx"]
if not len(split):
    print("You do not have any test data. Try changing your train-val-test split ratios.")
    sys.exit()

traj = connect(sys.argv[1])

test = []
for idx in split:
    idx = int(idx)
    atoms = traj.get_atoms(id=idx+1)
    test.append(atoms)

write("pbnn_test_data.traj", test)

