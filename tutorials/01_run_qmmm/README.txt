###################README###################
This tutorial teaches how to use the code to 
build a training dataset for the PBNN model.

Copy build_qm_mm_database.xml, diabat1_solvent.pdb
and plumed_command_0.txt to the parent directory
(containing main.py) and run

python main.py --xml_input build_qm_mm_database.xml

Currently this will only run for 5.0 fs for demonstration
purposes. Change the "num_steps" element under simulation settings
to run for a longer simulation time.

This simulation will produce a directory called "qmmm_output"
which we include an example of here. The colvar.dat
file lists the current value of the CV among other
info related to Plumed. The files

diabat1_monomer1.traj
diabat1_monomer2.traj
diabat2_monomer1.traj
diabat2_monomer2.traj

contains the coordinates for each monomer within the 
two diabats. We need to populate them with monomer
energies and forces for training the 
NNIntra terms in the Hamiltonian. Note that, in this
case, the diabats are symmetric. Also note that we
don't need a NNIntra term to predict the intramolecular
energy of Cl-. You can use the script "combine_traj.py"
in this directory to make one trajectory file containing
the training points for the Ch3Cl monomer.

python combine_traj.py qmmm_output/diabat1_monomer1.traj qmmm_output/diabat2_monomer2.traj ch3cl_monomer.traj

where the first two arguments to combine_traj.py are
trajectory files and the last argument is the name 
of the output file. With this done, copy ch3cl_monomer.traj
and the qmmm_intra_add.xml file to the base directory.
The command 

python main.py QMMM --xml_input qmmm_intra_add.xml

This will produce a file named qmmm_intra_output.traj.
This file is used in step 2 of this tutorial.

The file "reacting_complex.db" contains training data
for step 3 of this tutorial. It contains snapshots
of the reacting complex in an ASE database format along
with energies, forces, the external potential and the 
external field. 

sn2_liquid.dcd contains the trajectory and sn2_liquid.log
contains the energy log.
sn2_liquid.traj is a snapshot from the last recorded 
step of the trajectory. This can be used to restart
a QMMM simulation.
