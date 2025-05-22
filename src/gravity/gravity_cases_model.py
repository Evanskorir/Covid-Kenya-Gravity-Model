import numpy as np

from src.gravity.gravity_base_model import BaseGravityModel


class CasesGravityModel(BaseGravityModel):
    def __init__(self, data_loader, distances_from_nairobi, alpha=0.05, output_dir="output"):
        features = [
            "log_population", "log_gdp", "log_working", "log_distance",
            "poverty_rate", "internet_access", "tv_access",
            "population_tested", "positive_test_rates", "population_vaccinated"
        ]
        super().__init__(data_loader, distances_from_nairobi, "cases",
                         features, alpha, output_dir)

    def _extract_features(self, county, distance, _):
        try:
            P = self.data.population[county]
            G = self.data.gdp[county]
            W = self.data.working_population[county]
            PR = self.data.poverty_rate[county]
            TV = self.data.tv_access[county]
            NET = self.data.internet_usage[county]
            PT = self.data.pop_tested[county]
            PTR = self.data.positive_tested_rates[county]
            PV = self.data.pop_vaccinated[county]

            if any(v is None or v <= 0 for v in [P, G, W, distance]):
                return None

            return {
                "log_population": np.log(P + 1e-6),
                "log_gdp": np.log(G + 1e-6),
                "log_working": np.log(W + 1e-6),
                "log_distance": np.log(distance + 1e-6),
                "poverty_rate": PR / 100.0,
                "internet_access": NET / 100.0,
                "tv_access": TV / 100.0,
                "population_tested": PT / 100.0,
                "positive_test_rates": PTR / 100.0,
                "population_vaccinated": PV / 100.0,
            }

        except KeyError:
            return None
