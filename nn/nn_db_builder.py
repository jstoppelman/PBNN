import mdtraj as md
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase import db
import os
import numpy as np

class NNDataBaseBuilder:
    """
    Get info from the QMMM class in order to save frames of a simulation
    """
    def __init__(self, pdb_files, graph_reorder, qm_atoms, sim_dir, save_frq, write_mode):
        """
        pdb_files : list
            List of pdb files for the diabats in PBNN
        graph_reorder : class
            Reorder atoms for saving diabat 2 monomer positions
        qm_atoms : list
            Indices of atoms in QM region
        sim_dir : str
            Location to save DB files
        save_frq : int
            Save coordinates every n frames during a simulation
        write_mode : str
            whether to read or write frames
        """
        self.pdb_files = pdb_files
        self.graph_reorder = graph_reorder
        self.qm_atoms = qm_atoms
        self.sim_dir = sim_dir
        self.save_frq = save_frq
        self.write_mode = write_mode

        self._get_monomer_indices()

        if not os.path.isdir(f"{self.sim_dir}"): os.mkdir(f"{self.sim_dir}")
        if os.path.isfile(f"{self.sim_dir}/reacting_complex.db") and write_mode == "w": os.remove(f"{self.sim_dir}/reacting_complex.db")
        self.reacting_complex_database = db.connect(f"{self.sim_dir}/reacting_complex.db")

        self.monomer_writers = []
        for i, diabat in enumerate(self.diabat_monomer_indices):
            diabat_writers = []
            for j, monomer in enumerate(diabat):
                name = f"diabat{i+1}_monomer{j+1}"
                writer = TrajectoryWriter(f"{self.sim_dir}/{name}.traj", mode=self.write_mode)
                diabat_writers.append(writer)
            self.monomer_writers.append(diabat_writers)

    def _get_monomer_indices(self):
        """
        Get indices of the monomers in each diabat
        """
        self.diabat_monomer_indices = []
        for pdb in self.pdb_files:
            diabat_monomers = []
            traj = md.load(pdb)
            traj = traj.atom_slice(self.qm_atoms)
            for res in traj.top.residues:
                monomer = []
                for atom in res.atoms:
                    monomer.append(atom.index)
                diabat_monomers.append(monomer)

            self.diabat_monomer_indices.append(diabat_monomers)

    def write_db(self, atoms, frame, energy, forces, e_field=None, e_potential=None):
        """
        Write data to a DB file

        Parameters
        -----------
        atoms : object
            ASE atoms object
        frame : int
            Frame number
        energy : np.ndarray
            Total energy in kj/mol
        forces : np.ndarray
            Total forces in kj/mol
        e_field : optional, np.ndarray
            Contains external electric field
        e_potential : optional, np.ndarray
            Contains external electrical potential on each atom
        """

        if frame % self.save_frq == 0:
            reacting_complex = atoms[self.qm_atoms]
            reacting_complex_forces = forces[self.qm_atoms]
            
            calc = SinglePointDFTCalculator(reacting_complex, energy=energy, forces=reacting_complex_forces)
            reacting_complex.set_calculator(calc)
            #Determine if e_field is being saved
            if e_field is not None:
                e_field = np.asarray(e_field)
                e_potential = np.asarray(e_potential)
                self.reacting_complex_database.write(reacting_complex, data={'e_field': e_field, 'e_potential': e_potential})
            else:
                self.reacting_complex_database.write(reacting_complex)
            
            #Save monomer coordinates
            for i, diabat in enumerate(self.diabat_monomer_indices):
                diabat_writers = self.monomer_writers[i]
                for j, monomer in enumerate(diabat):
                    monomer_writer = diabat_writers[j]

                    if i == 1:
                        reacting_complex_tmp, indices = self.graph_reorder.reorder(reacting_complex)
                        atoms_monomer = reacting_complex_tmp[monomer]

                        reacting_complex_forces_tmp = reacting_complex_forces[indices]
                        forces_monomer = reacting_complex_forces_tmp[monomer]
                    else:
                        atoms_monomer = reacting_complex[monomer]
                        forces_monomer = reacting_complex_forces[monomer]

                    calc = SinglePointDFTCalculator(atoms_monomer, forces=forces_monomer)
                    atoms_monomer.set_calculator(calc)
                    monomer_writer.write(atoms=atoms_monomer)
            

