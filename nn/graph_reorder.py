#!/usr/bin/env python
import networkx as nx
from networkx.algorithms import isomorphism
from copy import deepcopy
import mdtraj as md
import numpy as np

class GraphReorder:
    """
    A class that creates graphs of the molecules in the primary diabat (whichever is classified as "diabat 1") and a secondary diabat. Without the reacting atoms, 
    the diabat graphs are isomorphic to one another. We use networkx tools to get the corresponding indices to morph the diabat 1 positions to the secondary diabat.
    IMPORTANT: this class should work for many cases, but it has only been tested in cases in which one or two atoms can accept the reacting group
    in the second diabat. It's important to test that the atom ordering is as desired for many cases.
    """
    def __init__(self, omm_topology, reacting_atoms, accepting_atoms, diabat_residues=None):
        """
        Parameters
        -----------
        omm_topology : list
            list of OpenMM topology objects for each diabat
        reacting_atoms : list
            list of ints: the indices of the atom for which the bond breaks/forms in each diabat
        accepting_atoms : list
            list of ints: the indices of the atoms in the secondary diabat which can accept the reacting group
        diabat_residues : list or None
            If there are solvent molecules present, we only need to build the graph for the reacting residues in the diabat
        """
       
        self.diabats = []
        #Form ForceBalance Molecule objects for each pdb file. The Molecule object forms a networkx graph for each detected molecule in the PDB file
        
        for top in omm_topology:
            md_top = md.Topology.from_openmm(top)
            #MDTraj can make networkx graphs from a topology
            graph = md_top.to_bondgraph()
            graph = nx.convert_node_labels_to_integers(graph)
            subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            self.diabats.append(subgraphs)

        self.reacting_atoms = reacting_atoms
        self.accepting_atoms = accepting_atoms
        self.diabat_residues = diabat_residues

        self._setup_graphs()

    def _setup_graphs(self):
        """
        Build molecule graphs and determine reordering from one to the other
        """
       
        #Combine the two molecule graphs in each Molecule object
        self.diabats_nonreactive_atom = []
        for diabat in self.diabats:
            graph = nx.empty_graph()
            if self.diabat_residues:
                for mol in self.diabat_residues:
                    graph = nx.compose(graph, diabat[mol])
            else:
                for mol in range(len(diabat)):
                    graph = nx.compose(graph, diabat[mol])
            self.diabats_nonreactive_atom.append(graph)

        #The reacting atom ruins the isomorphism between the two graphs. Remove the reacting atom in each diabat in order for NetworkX to detect this isomorphism
        for i, graph in enumerate(self.diabats_nonreactive_atom):
            for atom in self.reacting_atoms[i]:
                graph.remove_node(atom)
       
        #Checks whether there is an isomorphism between graph 0 and 1.
        graph_match = isomorphism.GraphMatcher(self.diabats_nonreactive_atom[0], self.diabats_nonreactive_atom[1])
        graph_match.is_isomorphic()
       
        #The mapping lists the indices in diabat 1 and the corresponding index value in diabat 2
        self.mapping = graph_match.mapping
        #Add back in the reacting atoms to the dictionary
        for atom1, atom2 in zip(self.reacting_atoms[0], self.reacting_atoms[1]):
            self.mapping[atom1] = atom2
        #Sort the mapping dictionary by diabat 1 index
        self.mapping = dict(sorted(self.mapping.items()))

        #Form the reverse mapping to go from diabat 2 to diabat 1
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

        #Create a list that can be used to reorder the atoms object
        self.reorder_list = np.zeros((max(list(self.mapping.keys()))+1))
        for k, v in self.reverse_mapping.items():
            self.reorder_list[k] = v

        self.diabat_reorder = deepcopy(self.reorder_list)

        #The accepting atoms are listed in their diabat 2 order. Remake this using the reverse mapping dictionary so that they are in diabat 1 order
        self.accepting_atoms_d1 = []
        for atom in self.accepting_atoms:
            self.accepting_atoms_d1.append(self.reverse_mapping[atom])
        self.accepting_atoms_d1 = np.asarray(self.accepting_atoms_d1)
      
        #Build the reorder list so that it includes nonreactive residues as well
        #If diabat_residues[0] isn't residue 0, then put in nodes of nonreactive atoms in
        if self.diabat_residues[0] != 0:
            for g in range(self.diabat_residues[0]):
                nodes_d1 = sorted(self.diabats[0][g].nodes)
                nodes_d2 = sorted(self.diabats[1][g].nodes)
                self.reorder_list[nodes_d2] = nodes_d1

        #If residues separate diabat_residue[0] and diabat_residue[1], get the indices of these
        #residues into reorder_list
        for g in range(self.diabat_residues[0], self.diabat_residues[1]):
            if g not in self.diabat_residues:
                nodes_d1 = sorted(self.diabats[0][g].nodes)
                nodes_d2 = sorted(self.diabats[1][g].nodes)
                self.reorder_list[nodes_d2] = nodes_d1
        
        #do the same thing for the other residues
        self.reorder_list = self.reorder_list.astype(int)
        self.reorder_list = self.reorder_list.tolist()
        for g in range(self.diabat_residues[1]+1, len(self.diabats[0])):
            nodes_d2 = sorted(self.diabats[1][g].nodes)
            for n in nodes_d2: 
                self.reorder_list.append(n)

    def reorder(self, atoms):
        """
        Parameters
        -----------
        atoms : ASE Atoms object
            atoms object containing the dimer data
        
        Returns
        -----------
        new_atoms : ASE Atoms object
            atoms object newly reordered
        reorder_list : list
            list used to form the reordered atoms object
        """

        #The atoms ordering created by networkx may not be correct. For example, in a carboxylic acid a proton could bond to either oxygen atom.
        #Create a test new_atoms object and determine if the distance to the reacting group from the multiple potential acceptors is the same as in NetworkX.
        new_atoms = atoms[self.reorder_list]
        #Compute distances using ASE
        if len(self.accepting_atoms) > 1:
            dists = new_atoms.get_distances(self.reacting_atoms[1], self.accepting_atoms, mic=True)
            a_min = np.argmin(dists)
            #If the shortest accepting atom/reacting atom distance is not 0, then create a new ordering
            if a_min != 0:
                accepting_atoms = np.asarray(self.accepting_atoms)
                #Create accepting atoms in diabat 2 in reverse order (largest distance to smallest distance)
                accepting_atoms = accepting_atoms[
                        np.argsort(dists)[::-1]
                        ].astype(int).tolist()
                #Create accepting_atoms_d1 (smallest distance to largest distance)
                accepting_atoms_d1 = self.accepting_atoms_d1[
                        np.argsort(dists)
                        ].astype(int).tolist()

                #Create new mapping dictionary
                mapping = deepcopy(self.mapping)
                #The accepting atoms in diabat 1 must correspond to different atoms in diabat 2 than in the original mapping
                for i, atom in enumerate(accepting_atoms):
                    mapping[accepting_atoms_d1[i]] = atom
               
                #Create new list and atoms object with the updated map
                reverse_mapping = {v: k for k, v in mapping.items()}
                reorder_list = np.zeros((max(list(self.mapping.keys()))+1))
                for k, v in reverse_mapping.items():
                    reorder_list[k] = v
               
                if self.diabat_residues[0] != 0:
                    for g in range(self.diabat_residues[0]):
                        nodes_d1 = self.diabats[0][g].nodes
                        nodes_d2 = self.diabats[1][g].nodes
                        reorder_list[nodes_d2] = nodes_d1

                for g in range(self.diabat_residues[0], self.diabat_residues[1]):
                    if g not in self.diabat_residues:
                        nodes_d1 = self.diabats[0][g].nodes
                        nodes_d2 = self.diabats[1][g].nodes
                        reorder_list[nodes_d2] = nodes_d1

                reorder_list = reorder_list.astype(int)
                reorder_list = reorder_list.tolist()
                for g in range(self.diabat_residues[1]+1, len(self.diabats[0])):
                    nodes_d2 = self.diabats[1][g].nodes
                    for n in nodes_d2: reorder_list.append(n)

                new_atoms = atoms[reorder_list]
                return new_atoms, reorder_list
        
        else:
            
            return new_atoms, self.reorder_list
