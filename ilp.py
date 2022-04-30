""" ILP Solver """

import os
# os.environ["XPAUTH_PATH"] = "/home/dajwani/software/licenses/xpauth_8_9_20.xpr"

import xpress as xp
import numpy as np
import pickle

def ilp_problem(adj_matrix):
    N, M = adj_matrix.shape # number squares, number lines
    x = np.array([xp.var(vartype = xp.binary) for i in range(M)])
    
    count_selected_sets = xp.Sum(x)

    validity_constraint = [ xp.Dot(x, adj_matrix.T) >= 1 ]

    p = xp.problem()
    p.addVariable(x)
    p.addConstraint(validity_constraint)
    p.setObjective(count_selected_sets, sense=xp.minimize)
    
    p.solve()
    p.getObjVal() 
    return p.getSolution(), p.getObjVal()