"""
this model is based on paper "An adaptive large neighborhood search heuristic for the
Pollution-Routing Problem", Emrah Demir, Tolga Bektas, Gilbert Laporte
"""
#%%
import math

import gurobipy as gp
from gurobipy import GRB

from utils import read_instance
from constants import *
from PR_Problem import PRProblem


def compute_objective(instance:PRProblem, x_vars, z_vars, f_vars, s_vars):
    const1 = FRICTION_FACTOR * ENGINE_SPEED * ENGINE_DISPLACEMENT * LAMBDA
    const2 = CURB_WEIGHT * GAMMA * LAMBDA * ALPHA
    const3 = GAMMA * LAMBDA * ALPHA
    const4 = BETTA * GAMMA * LAMBDA

    objective = 0
    for i in range(1, len(instance.customers)):
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


def constraint_12(model, instance:PRProblem, x_vars):
    fleet_size = len(instance.max_payload)
    _sum = 0
    for j in range(1, len(instance.customers)):
        _sum += x_vars[(0, j)]
    
    constraint_12 = "constraint_12_{}".format(j)
    model.addConstr(_sum == fleet_size, constraint_12)


def constraint_13_14(model, instance:PRProblem, x_vars):
    for i in range(1, len(instance.customers)):
        _sum1 = 0
        _sum2 = 0
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum1 += x_vars[(i, j)]
                _sum2 += x_vars[(j, i)]
        
        constraint_13 = "constraint_13_{}".format(i)
        constraint_14 = "constraint_14_{}".format(i)
        model.addConstr(_sum1 == 1, constraint_13)
        model.addConstr(_sum2 == 1, constraint_14)


def constraint_15(model, instance:PRProblem, f_vars):
    for i in range(1, len(instance.customers)):
        q = instance.customers[i]["demand"]
        _sum1 = 0
        _sum2 = 0
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum1 += f_vars[(j, i)]
                _sum2 += f_vars[(i, j)]

        constraint_15 = "constraint_15_{}".format(i)
        model.addConstr(_sum2 - _sum1 == q, constraint_15)


def constraint_16(model, instance: PRProblem,  x_vars, f_vars):
    for i in range(1, len(instance.customers)):
        Q = 10**9
        q = 0
        _sum1 = 0
        for j in range(1, len(instance.customers)):
            q = instance.customers[j]["demand"]
            if i != j:
                _sum1 = q * x_vars[(i, j)]

                constraint_16_1 = "constraint_16_1_{}_{}".format(i, j)
                constraint_16_2 = "constraint_16_2_{}_{}".format(i, j)
                model.addConstr(_sum1 <= f_vars[(i, j)], constraint_16_1)
                model.addConstr(f_vars[(i, j)] <= (Q - q) * x_vars[(i, j)], constraint_16_2)


def constraint_17(model, instance: PRProblem,  y_vars, z_vars, x_vars):
    K = 10**9
    for i in range(1, len(instance.customers)):
        t = instance.customers[i]["service_time"]
        _sum = 0
        for j in range(1, len(instance.customers)):
            if i != j:
                _sum = y_vars[i] - y_vars[j] + t 
                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += instance.dist[(i, j)] * z_vars[(i, j, r)] / r   

                mult = K*(1 - x_vars[(i, j)])
                constraint_17 = "constraint_17_{}_{}".format(i, j)
                model.addConstr(_sum <= mult, constraint_17)


def constraint_18(model, instance: PRProblem,  y_vars):
    for i in range(1, len(instance.customers)):
        a = instance.customers[i]["ready_time"]
        b = instance.customers[i]["due_time"]

        constraint_id_18_1 = "constraint_18_1_{}".format(i)
        constraint_id_18_2 = "constraint_18_2_{}".format(i)
        model.addConstr(a <= y_vars[i], constraint_id_18_1)
        model.addConstr(y_vars[i] <= b, constraint_id_18_2)


def constraint_19(model, instance: PRProblem,  x_vars, z_vars, y_vars, s_vars):
    L = 10**9
    for j in range(1, len(instance.customers)):
        t = instance.customers[j]["service_time"]
        _sum = y_vars[j] + t - s_vars[j]
        for r in range(instance.min_speed, instance.max_speed + 1):
            _sum += instance.dist[(j, 0)] * z_vars[(j, 0, r)] / r

        constraint_19 = "constraint_19_{}".format(j)
        model.addConstr(_sum == L*(1 - x_vars[(j, 0)]), constraint_19)


def constraint_20(model, instance:PRProblem, x_vars, z_vars):
    for i in range(0, len(instance.customers)):
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum = 0
                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += z_vars[(i, j, r)]

                constraint_20 = "constraint_20_{}_{}_{}".format(i, j, r)
                model.addConstr(_sum == x_vars[(i, j)], constraint_20)


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
    constraint_12(model, instance, x_vars)
    constraint_13_14(model, instance, x_vars)
    constraint_15(model, instance, f_vars)
    constraint_16(model, instance,  x_vars, f_vars)
    constraint_17(model, instance,  y_vars, z_vars, x_vars)
    constraint_18(model, instance,  y_vars)
    constraint_19(model, instance, x_vars, z_vars, y_vars, s_vars)
    constraint_20(model, instance, x_vars, z_vars)

    model.setParam(GRB.Param.IntFeasTol, 10**-4)
    model.optimize()

instance = read_instance(inst_name="UK10_01")
build_model(instance)
