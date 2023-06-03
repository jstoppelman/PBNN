# PBNN
[![doi:10.1063/5.0063187](https://img.shields.io/badge/DOI-10.1063%2F5.0063187-blue)](https://doi.org/10.1063/5.0063187)

PBNN is a reactive molecular dynamics code designed to simulate reactions in the condensed phase. The Hamiltonian is based on the EVB model and uses a combination of physics-based force field terms (called with OpenMM) and neural networks (called with PyTorch). The package can currently be used to run QM/MM as well, and has features available for training the PBNN Hamiltonian to QM/MM data (see the Tutorial directory).

# Table of contents
- [Install](#install)

# Install
To install, there are a few packages that need to be loaded. Make sure you have Python >=3.9.0. Then run the following commands
```
pip install -r requirements.txt
conda install -c conda-forge openmm
conda install -c conda-forge ase
```

Next, intall [SchNetPack](https://github.com/jstoppelman/schnetpack) from my own repo.
Finally, install [Psi4](https://github.com/johnppederson/psi4) to run QM/MM. You can find instructions for installing Psi4 in the docs directory I have included in this package.
