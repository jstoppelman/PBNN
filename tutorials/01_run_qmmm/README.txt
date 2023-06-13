###################README###################
This tutorial teaches how to use the code to 
build a training dataset for the PBNN model.

Copy both build_qm_mm_database.xml and 
diabat1_solvent.pdb to the parent directory
and run

python main.py --xml_input build_qm_mm_database.xml

This will produce a directory called "sn2_liquid"
This directory contains 4 .traj files (ASE trajectory
format) which have the positions of one of the monomers
in each diabat. These can be fed through the main
program to produce energies and forces for each monomer
or the user can run the Psi4 calculations themselves.
The last file is called "reacting_complex.db". This
contains the positions, the energy and forces from
Psi4 and the external potential and field on each
atom.

