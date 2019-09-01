
class Results(object):
    def __init__(self):
        self.solver_wallclock_time = None
        self.total_wallclock_time = None
        self.termination_condition = None
        self.objective_value = None
        self.lower_bound = None
        self.upper_bound = None
        self.duals = None
        self.reduced_costs = None
        self.slacks = None
