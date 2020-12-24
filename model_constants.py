import math

EMISSION_COST = 1.4  # f_c
FRICTION_FACTOR = 0.2  # k
ENGINE_SPEED = 33  # N
ENGINE_DISPLACEMENT = 5  # V
LAMBDA = 1 / (44 * 737)  # xi/(kappa*phi), xi = 1, kappa = 44, phi = 737
CURB_WEIGHT = 6350  # w
GAMMA = 1 / (1000 * 0.4 * 0.9)  # 1/(1000*n_tf*n)
BETTA = 0.5 * 0.7 * 1.2041 * 3.912  # 0.5*Cd*Rho*A
DRIVER_COST = 0.0022  # f_d (pounds/s)
ROLLING_RESISTANCE = 0.01  # C_r
GRAVITY = 9.81  # g
fixed_degree = 0.0107459922
ALPHA = GRAVITY * (math.sin(fixed_degree) + ROLLING_RESISTANCE * math.cos(fixed_degree))


