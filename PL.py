"""
this model is based on paper "An adaptive large neighborhood search heuristic for the
Pollution-Routing Problem", Emrah Demir, Tolga Bektas, Gilbert Laporte
"""

import math

import gurobipy as gp
from gurobipy import GRB

from PRP.utils import read_instance
from PRP.constants import *
from PRP.PR_Problem import PRProblem


def compute_objective(instance:PRProblem, x_vars, z_vars, f_vars, s_vars):
    const1 = FRICTION_FACTOR * ENGINE_SPEED * ENGINE_DISPLACEMENT * LAMBDA
    const2 = CURB_WEIGHT * GAMMA * LAMBDA * ALPHA
    const3 = GAMMA * LAMBDA * ALPHA
    const4 = BETTA * GAMMA * LAMBDA

    objective = 0
    for i in range(0, len(instance.customers)):
        for j in range(0, len(instance.customers)):
            if i != j:
                sum_z = 0
                sum_z2 = 0
                for r in range(instance.min_speed, instance.max_speed + 1):
                    sum_z += z_vars[(i, j, r)] / r
                    sum_z2 += z_vars[(i, j, r)] * (r ** 2)
                objective += const1 * instance.dist[(i, j)] * sum_z
                objective += const2 * instance.dist[(i, j)] * x_vars[(i, j)]
                objective += const3 * instance.dist[(i, j)] * f_vars[(i, j)]
                objective += const4 * instance.dist[(i, j)] * sum_z2
                objective += DRIVER_COST * s_vars[i]
    return objective


def constraint_13(model, instance:PRProblem, x_vars):
    fleet_size = len(instance.max_payload)
    _sum = 0
    for j in range(1, len(instance.customers)):
        _sum += x_vars[(0, j)]
    model.addConstr(_sum == fleet_size, "constraint_13")


def constraint_14_15(model, instance:PRProblem, x_vars):
    for i in range(1, len(instance.customers)):
        _sum1 = 0
        _sum2 = 0
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum1 += x_vars[(i, j)]
                _sum2 += x_vars[(j, i)]

        model.addConstr(_sum1 == 1, "constraint_14")
        model.addConstr(_sum2 == 1, "constraint_15")


def constraint_20(model, instance: PRProblem,  x_vars, z_vars, y_vars, s_vars):
    L = math.inf
    for j in range(1, len(instance.customers)):
        t = instance.customers[j]["service_time"]
        _sum = y_vars[j] + t - s_vars[j]
        for r in range(instance.min_speed, instance.max_speed + 1):
            _sum += instance.dist[(j, 0)] * z_vars[(j, 0, r)] / r

        model.addConstr(_sum == L*(1 - x_vars[(j, 0)]), "constraint_20")


def constraint_21(model, instance:PRProblem, x_vars, z_vars):
    for i in range(0, len(instance.customers)):
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum = 0
                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += z_vars[(i, j, r)]

                model.addConstr(_sum == x_vars[(i, j)], "constraint_21")


def build_model(instance: PRProblem):
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
        for j in range(0, len(instance.customers)):
            if i != j:
                x_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name="x_{}{}".format(i, j))
                f_vars[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_{}{}".format(i, j))
                for r in range(instance.min_speed, instance.max_speed + 1):
                    z_vars[(i, j, r)] = model.addVar(vtype=GRB.BINARY, name="z_{}{}{}".format(i, j, r))

    objective = compute_objective(instance, x_vars, z_vars, f_vars, s_vars)
    model.setObjective(objective, GRB.MINIMIZE)

    # adding constraints
    constraint_13(model, instance, x_vars)
    constraint_14_15(model, instance, x_vars)
    constraint_20(model, instance, x_vars, z_vars, y_vars, s_vars)
    constraint_21(model, instance, x_vars, z_vars)

instance = read_instance(inst_name="UK10_01")

