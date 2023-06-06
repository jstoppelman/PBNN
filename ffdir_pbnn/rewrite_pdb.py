#!/usr/bin/env python
from MDAnalysis import *
import numpy as np

u = Universe("diabat1_solvent.pdb")

resnums = np.arange(1, len(u.residues.resnums)+1)
resid = np.arange(1, len(u.residues.resnums)+1)

#Make empty universe object "cg" containing only the center of mass positions of each molecule.
cg = Universe.empty(n_atoms=len(u.atoms), n_residues=u.residues.n_residues,
                                atom_resindex=u.atoms.resindices, trajectory=True)
cg.add_TopologyAttr("resnames", u.residues.resnames)
cg.add_TopologyAttr("resnums", resnums)
cg.add_TopologyAttr("names", u.atoms.names)
cg.add_TopologyAttr("resid", resid)

#Coordinates variable will contain the xyz positions of the center of mass positions of each molecule
#(number of molecules given by u.residues.n_residues) for each frame (given by len(u.trajectory))
coordinates = np.empty((len(u.trajectory), len(u.atoms), 3))

frame = 0
add = 0
for ts in u.trajectory:
    #if frame % 100 == 0:
    coordinates[frame] = ts.positions
    frame += 1

#Load the coordinates into the cg universe
cg.load_new(coordinates, order='fac')

#Now load unitcell vectors from the complete trajectory u into cg
#Unitcell vectors are given by the "dimensions" attribute
dims = u.dimensions
for ts in cg.trajectory:
    ts.dimensions = dims

ag = cg.select_atoms("resid 0:2002")
ag.write("diabat1_solvent_out.pdb")
