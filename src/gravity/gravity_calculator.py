
class GravityModel:
    def __init__(self, cases_dict, population_dict, distance_calculator):
        self.cases = cases_dict
        self.population = population_dict
        self.distance_calculator = distance_calculator
        self.gravity_matrix = {}

    def compute_gravity(self):
        distances = self.distance_calculator.get_all_distances()

        for (origin, destination), D_ij in distances.items():
            if origin not in self.cases or destination not in self.population:
                continue

            C_i = self.cases[origin]
            P_j = self.population[destination]

            if D_ij == 0:
                continue  # avoid division by zero

            G_ij = (C_i * P_j) / (D_ij ** 2)
            self.gravity_matrix[(origin, destination)] = G_ij

    def get_gravity(self, origin, destination):
        return self.gravity_matrix.get((origin, destination))

    def get_all_gravity(self):
        return self.gravity_matrix

