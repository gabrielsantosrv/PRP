class PRProblem:
    def __init__(self, n_customers, vehicle_curb, max_payload, min_speed, max_speed, dist, customers,
                 fleet_size=None):
        self.n_customers = n_customers
        self.vehicle_curb = vehicle_curb
        self.max_payload = max_payload
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.dist = dist
        self.customers = customers
        self.fleet_size = fleet_size

    def __str__(self):
        return "n. customers: {}\n" \
               "vehicle curb: {}\n" \
               "max payload: {}\n" \
               "min speed: {} max speed: {}\n" \
               "distances: {}\n" \
               "customers: {}".format(self.n_customers,
                                      self.vehicle_curb,
                                      self.max_payload,
                                      self.min_speed, self.max_speed,
                                      self.dist,
                                      self.customers)
