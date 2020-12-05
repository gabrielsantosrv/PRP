import gurobipy as gp
from gurobipy import GRB

from utils import read_instance
from constants import *
from PRP import PRP


def build_model(instance:PRP):
    model = gp.Model("PRP")

    z_vars = {}
    x_vars = {}
    f_vars = {}
    y_vars = {}
    s_vars = {}
    for i in range(0, len(instance.customers)):
        if i > 0:
            y_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="y_{}".format(i))
            s_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="s_{}".format(i))
        for j in range(i+1, len(instance.customers)):
            x_vars[(i,j)] = model.addVar(vtype=GRB.BINARY, name="x_{}{}".format(i, j))
            f_vars[(i,j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_{}{}".format(i, j))
            for r in range(instance.min_speed, instance.max_speed + 1):
                z_vars[(i,j,r)] = model.addVar(vtype=GRB.BINARY, name="z_{}{}{}".format(i, j, r))

    objective = compute_objective(instance, x_vars, z_vars, f_vars, s_vars)
    model.setObjective(objective, GRB.MINIMIZE)



def compute_objective(instance:PRP, x_vars, z_vars, f_vars, s_vars):
    const1 = friction_factor * engine_speed * engine_displacement * lambda_value
    const2 = curb_weight * gamma * lambda_value * alpha
    const3 = gamma * lambda_value * alpha
    const4 = betta * gamma * lambda_value

    objective = 0
    for i in range(0, len(instance.customers)):
        for j in range(i+1, len(instance.customers)):
            sum_z = 0
            sum_z2 = 0
            for r in range(instance.min_speed, instance.max_speed + 1):
                sum_z += z_vars[(i, j, r)] / r
                sum_z2 += z_vars[(i, j, r)] * (r ** 2)
            objective += const1 * instance.dist[(i, j)] * sum_z
            objective += const2 * instance.dist[(i, j)] * x_vars[(i, j)]
            objective += const3 * instance.dist[(i, j)] * f_vars[(i, j)]
            objective += const4 * instance.dist[(i, j)] * sum_z2
            objective += driver_cost * s_vars[i]
    return objective

instance = read_instance(inst_name="UK10_01")

