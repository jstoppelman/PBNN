###################README###################
We can run MD simulations using the final 
trained models.
To run a gas phase simulation, copy
the files ch3cl2_0.xyz, pbnn_md_gas_phase.xml
and plumed_command_gas_phase.txt to the parent
directory.

python main.py PBNN --xml_input pbnn_md_gas_phase.xml

The output will be similar to QM/MM output. You can run
simulations with different plumed command files
in order to get a PMF.

The process for simulating the liquid phase
is similar. Copy pbnn_md_liquid.xml, diabat1_solvent.pdb 
and plumed_command_liquid.txt to the parent directory.

python main.py  --xml_input pbnn_md_liquid.xml
