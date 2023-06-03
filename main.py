#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from qm_mm import *
from read_input import *
from nn import *
import mdtraj as md
from ase import db

def main():
    """
    Run a simulation, set up databases for PB/NN neural network training, train the neural network or run simulations using the trained neural network
    """
    parser = argparse.ArgumentParser(description="""This code allows for either running a QM/MM simulation (currently with a cutoff), set up databases for training PB/NN, 
    train PB/NN or run a MD simulation using PB/NN""")
    parser.add_argument("jobtype", type=str, choices=["QMMM", "Train", "PBNN", "PBNN_Component"], help="List the jobtype that you would like to run")
    parser.add_argument("--xml_input", type=str, help="Enter xml file containing job options, see examples directory for input")

    args = parser.parse_args()

    jobtype = args.jobtype

    #Proceed based on which job type is described
    if jobtype == "QMMM":
        #Form an input reader for QMMM
        read_input = ReadSettingsQMMM(args.xml_input)
        
        #Get dictionary of settings
        settings = read_input.getSettings()

        #QMMMSetup object sets up ASE object
        qm_mm_setup = QMMMSetup(settings)

        #qm_mm is the ASE object, simulation_settings is one of the dictionaries produced from the xml input file 
        qm_mm, simulation_settings= qm_mm_setup.get_qmmm()
        name, ensemble, time_step, write_freq = simulation_settings["name"], simulation_settings["ensemble"], simulation_settings["time_step"], simulation_settings["write_freq"]
        
        num_steps = simulation_settings["num_steps"]

        #Gets friction for Langevin dynamics
        friction = simulation_settings.get("friction", None)

        temp, temp_init = simulation_settings["temp"], simulation_settings["temp_init"]

        remove_translation, remove_rotation = simulation_settings["remove_translation"], simulation_settings["remove_rotation"]

        #Sets up QM/MM MD simulation
        qm_mm.create_system(name, 
                ensemble=ensemble, 
                time_step=time_step, 
                write_freq=write_freq, 
                friction=0.005,
                temp=temp,
                temp_init=temp_init,
                remove_translation=remove_translation,
                remove_rotation=remove_rotation,
                embed_electrode=False)
        
        #Run simulation
        qm_mm.run_md(num_steps)

    #Used for training all NN models
    elif jobtype == "Train":
        #Read xml file for training settings
        read_input = ReadSettingsTraining(args.xml_input)
        #Get settings dictionary
        settings = read_input.getSettings()
        #Setup training object with settings dictionary
        train_setup = TrainSetup(settings)
        #Get trainer object
        trainer = train_setup.get_trainer()

        trainer.train()

    #Set up PBNN calculator for MD simulation or for testing PBNN against a dataset
    elif jobtype == "PBNN":
        read_input = ReadSettingsPBNN(args.xml_input)

        settings = read_input.getSettings()

        pbnn_setup = PBNNSetup(settings)
        pbnn = pbnn_setup.get_pbnn()

        if pbnn_setup.jobtype == "Test":
            pbnn.run_test()

        elif pbnn_setup.jobtype == "MD":
            num_steps = pbnn_setup.num_steps
            pbnn.run_md(num_steps)
    
    #Test a component of PBNN against a dataset (currently only works for singular NNIntra model)
    elif jobtype == "PBNN_Component":
        read_input = ReadSettingsPBNNComp(args.xml_input)

        settings = read_input.getSettings()

        pbnn_setup = PBNNComponent(settings)

        component_test = pbnn_setup.getTest()

        component_test.run_test()
        
if __name__ == "__main__":
    main()
