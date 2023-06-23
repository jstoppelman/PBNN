###################README###################
In this part of the tutorial, we parameterize
the remaining parts of the 2x2 Hamiltonian.
This includes the NNInter neural network
in the diagonal terms and the Hij off-diagonal
terms. The training data we need was set up
in step 1. Find where the simulation
directories were created in that step 
(the names were qmmm_output_gas and qmmm_output_liquid)
and cp the file combine_db.py

python combine_db.py qmmm_output_gas/reacting_complex_sn2_gas.db qmmm_output_liquid/reacting_complex_sn2_liquid.db ch3cl2_train.db

Make sure the intramolecular model is 
in the parent directory.
Then to train you can run the command
(assuming all needed files are in the
parent directory):
python main.py Train --xml_input qmmm_fit.xml

In the output directory lightning_logs_qmmm_fit, 
you will find the output files best_model_inter_d1,
best_model_inter_d2 and best_model_hij.
You can perform the same process
as in step 2 in order to view the 
training metrics in lightning_logs_qmmm_fit.

In order to test the performance of the 
total Hamiltonian, copy the file
get_test_set_db.py to the parent directory

python get_test_data_db.py ch3cl2_train.db

Also make sure you copy the best models 
to the parent directory.
You can then test the total PB/NN
Hamiltonian energy against what is in
the training data with 

python main.py PBNN --xml_input pbnn_singlepoint.xml

This will output npy files for which
you can test the energies and forces
against what is in the training set
with the MAE script from step 2.


