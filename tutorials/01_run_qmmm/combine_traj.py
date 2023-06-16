#!/usr/bin/env python
import argparse
from ase.io import read, write

parser = argparse.ArgumentParser(description="""Combine arbitrary number of ASE trajectories""")
parser.add_argument("traj_list", type=str, nargs="+", help="List of trajectory files")
parser.add_argument("output_name", type=str, help="Name of output file")

args = parser.parse_args()

traj_list = args.traj_list
output_name = args.output_name

total_traj = []
for f in traj_list:
    traj = read(f, index=":")
    for frame in traj: total_traj.append(frame)

write(output_name, total_traj)
