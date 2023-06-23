###################README###################
We can now use the traj file qmmm_intra_output.traj, 
which we constructed in the last step of the tutorial
in order to train the Intra NN term for CH3Cl.
From this directory, copy the xml file to the
parent directory

cp nnintra_train.xml ../../
cd ../../

Make sure the qmmm_intra_output.traj
file is in the same directory.
Then you can train the NN model with:

python main.py Train --xml_input nnintra_train.xml

The trained model will be located at 
"lightning_logs_intra/best_model".

We can copy and rename it to the parent directory
for the next step 

cp lightning_logs_intra/best_model ./ch3cl_model

PyTorch Lightning uses a tensorboard logging system
for storing training metrics like train MAE, validation
MAE etc. To get a text output, copy the script in this
directory:

cp convert_tboard.py ../../lightning_logs_nnintra/lightning_logs/version_0/
cd ../../

There will be a file with the prefix "events" in this 
directory. The name can be something like
"events.out.tfevents.1687463364.comp"
The numbers will change for each model trained.

Run the command

python convert_tboard.py events.out.tfevents.1687463364.comp

And you will get a csv file named output.csv
This contains basic info. You can look at 
the tensorboard documentation to explore 
other ways to view this file.

For testing the model on different structrues, 
you can use the file pbnn_component_intra.xml.
To get the test set from the total traj file,
copy the script

cp get_test_set.py ../../
cd ../../

to the parent directory.
Then you can run the command

python get_test_set.py qmmm_intra_output.traj

This will output a file named pbnn_test_data.traj
Copy the nnintra_singlepoint.xml file and run
the command

python main.py PBNN_Component --xml_input nnintra_singlepoint.xml

to output energies and forces from the NNIntra model (+ Morse potential
in this case) against the reference data contained 
in pbnn_test_data.traj. Running the get_mae.py script 
in here will print out the mean absolute error between  
the QM reference data and the NNIntra model + Morse potential
 
cp get_mae.py ../../
python get_mae.py
