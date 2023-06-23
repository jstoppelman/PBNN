#!/usr/bin/env python
import numpy as np
import sys
from ase.io import read, write

split = np.load("split.npz")["test_idx"]
if not len(split):
    print("You do not have any test data. Try changing your train-val-test split ratios.")
    sys.exit()

traj = read(sys.argv[1], index=":")

test = []
for idx in split: 
    test.append(traj[idx])

write("pbnn_test_data.traj", test)

