"""
this model is based on paper "An adaptive large neighborhood search heuristic for the
Pollution-Routing Problem", Emrah Demir, Tolga Bektas, Gilbert Laporte
"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
import math

from utils import read_instance
from model_constants import *
from PR_Problem import PRProblem

K = 10**9
L = 10**9

def compute_objective(instance:PRProblem, x_vars, z_vars, f_vars, s_vars):
    const1 = FRICTION_FACTOR * ENGINE_SPEED * ENGINE_DISPLACEMENT * LAMBDA
    const2 = CURB_WEIGHT * GAMMA * LAMBDA
    const3 = GAMMA * LAMBDA
    const4 = BETTA * GAMMA * LAMBDA
    alphas = genarateRandomAlphaMatrix()

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
                objective += const2 * ALPHA * instance.dist[(i, j)] * x_vars[(i, j)]
                objective += const3 * ALPHA * instance.dist[(i, j)] * f_vars[(i, j)]
                objective += const4 * instance.dist[(i, j)] * sum_z2
        if i > 0:
            objective += DRIVER_COST * s_vars[i]
    return objective


def constraint_12(model, instance:PRProblem, x_vars):
    _sum = 0
    for j in range(1, len(instance.customers)):
        _sum += x_vars[(0, j)]
    
    constraint_12_name = "constraint_12"
    model.addConstr(_sum == instance.fleet_size, constraint_12_name)


def constraint_13(model, instance:PRProblem, x_vars):
    for i in range(1, len(instance.customers)):
        _sum = 0
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum += x_vars[(i, j)]
        
        constraint_13_name = "constraint_13_{}".format(i)
        model.addConstr(_sum == 1, constraint_13_name)


def constraint_14(model, instance: PRProblem, x_vars):
    for j in range(1, len(instance.customers)):
        _sum = 0
        for i in range(0, len(instance.customers)):
            if i != j:
                _sum += x_vars[(i, j)]

        constraint_14_name = "constraint_14_{}".format(j)
        model.addConstr(_sum == 1, constraint_14_name)


def constraint_15(model, instance:PRProblem, f_vars):
    for i in range(1, len(instance.customers)):
        q = instance.customers[i]["demand"]
        _sum1 = 0
        _sum2 = 0
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum1 += f_vars[(j, i)]
                _sum2 += f_vars[(i, j)]

        constraint_15_name = "constraint_15_{}".format(i)
        model.addConstr(_sum1 - _sum2 == q, constraint_15_name)


def constraint_16(model, instance: PRProblem,  x_vars, f_vars):
    Q = instance.max_payload
    for i in range(0, len(instance.customers)):
        q_i = instance.customers[i]["demand"]
        for j in range(0, len(instance.customers)):
            q_j = instance.customers[j]["demand"]
            if i != j:
                constraint_16_1 = "constraint_16_1_{}_{}".format(i, j)
                constraint_16_2 = "constraint_16_2_{}_{}".format(i, j)
                model.addConstr(q_j * x_vars[(i, j)] <= f_vars[(i, j)], constraint_16_1)
                model.addConstr(f_vars[(i, j)] <= (Q - q_i) * x_vars[(i, j)], constraint_16_2)


def constraint_17(model, instance: PRProblem,  y_vars, z_vars, x_vars):
    for i in range(0, len(instance.customers)):
        t_i = instance.customers[i]["service_time"]
        for j in range(1, len(instance.customers)):
            _sum = 0
            if i != j:
                if i > 0:
                    _sum = y_vars[i] - y_vars[j] + t_i
                else:
                    _sum = - y_vars[j] + t_i

                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += instance.dist[(i, j)] * z_vars[(i, j, r)] / r   

                constraint_17_name = "constraint_17_{}_{}".format(i, j)
                model.addConstr(_sum <= K * (1 - x_vars[(i, j)]), constraint_17_name)

def constraint_17_indicator(model, instance: PRProblem,  y_vars, z_vars, x_vars):
    # reference: https://www.gurobi.com/documentation/8.0/refman/py_model_addgenconstrindic.html

    for i in range(0, len(instance.customers)):
        t_i = instance.customers[i]["service_time"]
        for j in range(1, len(instance.customers)):
            _sum = 0
            if i != j:
                if i > 0:
                    _sum = y_vars[i] + t_i
                else:
                    _sum = t_i
                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += instance.dist[(i, j)] * z_vars[(i, j, r)] / r

                model.addGenConstrIndicator(x_vars[(i, j)], True, _sum <= y_vars[j])

def constraint_18(model, instance: PRProblem,  y_vars):
    for i in range(1, len(instance.customers)):
        a = instance.customers[i]["ready_time"]
        b = instance.customers[i]["due_time"]

        constraint_id_18_1 = "constraint_18_1_{}".format(i)
        constraint_id_18_2 = "constraint_18_2_{}".format(i)
        model.addConstr(a <= y_vars[i], constraint_id_18_1)
        model.addConstr(y_vars[i] <= b, constraint_id_18_2)


def constraint_19(model, instance: PRProblem,  x_vars, z_vars, y_vars, s_vars):
    for j in range(1, len(instance.customers)):
        t_j = instance.customers[j]["service_time"]
        _sum = y_vars[j] + t_j - s_vars[j]
        for r in range(instance.min_speed, instance.max_speed + 1):
            _sum += instance.dist[(j, 0)] * z_vars[(j, 0, r)] / r

        constraint_19_name = "constraint_19_{}".format(j)
        model.addConstr(_sum <= L * (1 - x_vars[(j, 0)]), constraint_19_name)


def constraint_19_indicator(model, instance: PRProblem,  x_vars, z_vars, y_vars, s_vars):
    for j in range(1, len(instance.customers)):
        t_j = instance.customers[j]["service_time"]
        _sum = y_vars[j] + t_j
        for r in range(instance.min_speed, instance.max_speed + 1):
            _sum += instance.dist[(j, 0)] * z_vars[(j, 0, r)] / r

        model.addGenConstrIndicator(x_vars[(j, 0)], True, _sum <= s_vars[j])


def constraint_20(model, instance:PRProblem, x_vars, z_vars):
    for i in range(0, len(instance.customers)):
        for j in range(0, len(instance.customers)):
            if i != j:
                _sum = 0
                for r in range(instance.min_speed, instance.max_speed + 1):
                    _sum += z_vars[(i, j, r)]

                constraint_20_name = "constraint_20_{}_{}".format(i, j)
                model.addConstr(_sum == x_vars[(i, j)], constraint_20_name)


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
                x_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name="x_{}_{}".format(i, j))
                f_vars[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_{}_{}".format(i, j))
                for r in range(instance.min_speed, instance.max_speed + 1):
                    z_vars[(i, j, r)] = model.addVar(vtype=GRB.BINARY, name="z_{}_{}_{}".format(i, j, r))
    instance.fleet_size = model.addVar(vtype=GRB.INTEGER, lb=0, name="m")
    objective = compute_objective(instance, x_vars, z_vars, f_vars, s_vars)
    model.setObjective(objective, GRB.MINIMIZE)

    # adding constraints
    constraint_12(model, instance, x_vars)
    constraint_13(model, instance, x_vars)
    constraint_14(model, instance, x_vars)
    constraint_15(model, instance, f_vars)
    constraint_16(model, instance,  x_vars, f_vars)
    # constraint_17(model, instance,  y_vars, z_vars, x_vars)
    constraint_17_indicator(model, instance, y_vars, z_vars, x_vars)
    constraint_18(model, instance,  y_vars)
    # constraint_19(model, instance, x_vars, z_vars, y_vars, s_vars)
    constraint_19_indicator(model, instance, x_vars, z_vars, y_vars, s_vars)
    constraint_20(model, instance, x_vars, z_vars)

    return model, x_vars, z_vars

def genarateRandomAlphaMatrix():
    alphas = {}

    for i in range(0, len(instance.customers)):
        for j in range(0, len(instance.customers)):
            fixedDegree = 0.0107459922
            alphas[(i, j)] = GRAVITY * (math.sin(fixedDegree) + ROLLING_RESISTANCE * math.cos(fixedDegree))

    return alphas

def save_results_CSV(lower_bounds:list, upper_bounds:list, times:list, fleet_sizes:list, filepath):
    df = pd.DataFrame({'time':times, 'lower_bounds':lower_bounds, 'upper_bounds':upper_bounds, 'fleet size':fleet_sizes})
    df.to_csv(filepath)

if __name__ == '__main__':
    random.seed(42)
    lower_bounds = []
    upper_bounds = []
    times = []
    fleet_sizes = []
    for n in [100]:
        for i in range(1, 2):
            if i < 10:
                instance_name = "{}{}".format("UK{}_0".format(n), i)
            else:
                instance_name = "{}{}".format("UK{}_".format(n), i)
            print(instance_name)
            instance = read_instance(inst_name=instance_name)

            model, x_vars, z_vars = build_model(instance)
            model.setParam('TimeLimit', 1800)
            model.optimize()
            print('Final Objective: {}'.format(model.objVal))
            for v in model.getVars():
                if v.varName == "m":
                    print('Fleet size: %g' % (v.x))
                    fleet_sizes.append(v.x)

            print('Route')
            for edge, v in x_vars.items():
                if v.X > 0.9:
                    for r in range(instance.min_speed, instance.max_speed + 1):
                        if z_vars[(edge[0], edge[1], r)].X > 0.9:
                            print("{}: {}".format(v.varName, v.X), "velocidade:", r)

            lower_bounds.append(model.objBound)
            upper_bounds.append(model.objVal)
            times.append(model.runtime)

        # filepath = "./pl_{}_results.csv".format(n)
        # save_results_CSV(lower_bounds, upper_bounds, times, fleet_sizes, filepath)