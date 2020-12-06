class PRProblem:
    def __init__(self, n_customers, max_payload, min_speed, max_speed, dist, customers):
        self.n_customers = n_customers
        self.max_payload = max_payload
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.dist = dist
        self.customers = customers

    def __str__(self):
        return "n. customers: {}\n" \
               "max payload: {}\n" \
               "min speed: {} max speed: {}\n" \
               "distances: {}\n" \
               "customers: {}".format(self.n_customers,
                                      self.max_payload,
                                      self.min_speed, self.max_speed,
                                      self.dist,
                                      self.customers)
