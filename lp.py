""" LP Solver """

import xpress as xp
import numpy as np

xp.controls.outputlog = 0
xp.setOutputEnabled(False)

def lp_solver(set_cover, set_uncover, SWAP):
    R = len(set_cover) # number lines

    R_prime = len(set_uncover) # number of disks not selected

    M = R + R_prime # number of disks
    
    x = np.array([xp.var(vartype = xp.continuous, lb=0, ub=1) for i in range(R)])
    y = np.array([xp.var(vartype = xp.continuous, lb=0, ub=1) for i in range(R_prime)])
 

    removed_constraint = xp.Sum(x) == SWAP["remove"]
    selected_constraint = xp.Sum(y) == SWAP["select"]
    # print("in lp solver")

    validity_constraint = [ set_cover.sum() - xp.Dot(x, set_cover) + xp.Dot(y, set_uncover) >= 1]
    # print(validity_constraint)

    p = xp.problem()
    p.addVariable(x,y)
    p.addConstraint(removed_constraint, selected_constraint, validity_constraint)
    
    p.solve()
    # print("Solution: " + str(p.getSolution()))
    sol = p.getSolution()
    objVal = p.getObjVal()

    # print(len(sol))
    return p.getSolution(x), p.getSolution(y)
