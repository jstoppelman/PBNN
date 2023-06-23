#!/usr/bin/env python
import argparse
from ase.db import connect
import os

parser = argparse.ArgumentParser(description="""Combine arbitrary number of ASE trajectories""")
parser.add_argument("traj_list", type=str, nargs="+", help="List of trajectory files")
parser.add_argument("output_name", type=str, help="Name of output file")

args = parser.parse_args()

traj_list = args.traj_list
output_name = args.output_name

if os.path.isfile(output_name): os.remove(output_name)

output = connect(output_name)

for f in traj_list:
    traj = connect(f)
    for i in range(1, len(traj)+1):
        frame = traj.get(id=i)
        output.write(frame)
