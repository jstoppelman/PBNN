#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions accessed by multiple classes.
"""
import numpy as np


def least_mirror_distance(i_vector, j_vector, box):
    """
    Returns the least mirror convention distance between i_vec and
    j_vec.

    Parameters
    ----------
    i_vector: NumPy array
        First position vector
    j_vector: NumPy array
        Second position vector
    box: list of list of float
        Cell object from ASE, which contains the box vectors.

    Returns
    -------
    r_vector : NumPy array
        Least mirror vector between the first and second position
        vectors.
    """
    r_vector = [j_vector[k] - i_vector[k] for k in range(3)]
    r_vector -= box[2]*np.floor(r_vector[2]/box[2][2] + 0.5)
    r_vector -= box[1]*np.floor(r_vector[1]/box[1][1] + 0.5)
    r_vector -= box[0]*np.floor(r_vector[0]/box[0][0] + 0.5)
    return r_vector
