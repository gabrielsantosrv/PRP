from PR_Problem import PRProblem

def read_instance(inst_name):
    with open("instances/{}.txt".format(inst_name)) as file:
        n_customers = int(file.readline())
        file.readline()
        curb_payload = file.readline().split()
        vehicle_curb = float(curb_payload[0])
        max_payload = float(curb_payload[1])
        file.readline()
        speeds = file.readline().split()
        min_speed = int(speeds[0])
        max_speed = int(speeds[1])
        file.readline()
        dist = {}
        for row in range(0, n_customers+2):
            line = file.readline()
            for col, value in enumerate(line.split()):
                dist[(int(row), int(col))] = float(value)

        file.readline()
        customers = []
        while len(customers) != n_customers+1:
            line = file.readline()
            info = line.split()
            if len(info) > 0:
                customers.append({"id": int(info[0]),
                                  "city_name": info[1],
                                  "demand": float(info[2]),
                                  "ready_time": float(info[3]),
                                  "due_time": float(info[4]),
                                  "service_time": float(info[5])})

    return PRProblem(n_customers, vehicle_curb, max_payload, min_speed, max_speed, dist, customers)
