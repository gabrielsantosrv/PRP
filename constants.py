emission_cost = 1.4 # f_c
friction_factor = 0.2 # k
engine_speed = 33 # N
engine_displacement = 5 # V
lambda_value = 737/44
curb_weight = 6350 # w
gamma = 1/(1000*0.4*0.9) # 1/(1000*n_tf*n)
betta = 0.5*0.7*1.2041*3.912 # 0.5*Cd*Rho*A
driver_cost = 0.0022 # pounds/s
alpha = 1


const1 =  friction_factor * engine_speed * engine_displacement * lambda_value
