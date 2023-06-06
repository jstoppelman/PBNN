#!/usr/bin/env python
import parmed as pmd

traj = pmd.load_file("ch3cl_liquid_10_qm.pdb")
traj_mm = pmd.load_file("ch3cl_liquid_10_mm.pdb")

traj = traj + traj_mm


traj.save("ch3cl_liquid.pdb")
