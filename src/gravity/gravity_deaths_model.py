import numpy as np

from src.gravity.gravity_base_model import BaseGravityModel


class DeathsGravityModel(BaseGravityModel):
    def __init__(self, data_loader, distances_from_nairobi, alpha=0.05, output_dir="output"):
        features = [
            "log_gdp", "log_population", "log_distance", "any_insurance",
            "poverty_rate", "positive_test_rates",
            "NHIF_cover", "population_vaccinated", "elderly_persons"
        ]
        super().__init__(data_loader, distances_from_nairobi, "deaths",
                         features, alpha, output_dir)

    def _extract_features(self, county, distance, _):
        try:
            P = self.data.population[county]
            G = self.data.gdp[county]
            PR = self.data.poverty_rate[county]
            EP = self.data.elderly_persons[county]
            PTR = self.data.positive_tested_rates[county]
            PV = self.data.pop_vaccinated[county]
            AI = self.data.any_health_insurance[county]
            NHF = self.data.nhif[county]

            if any(v is None or v <= 0 for v in [P, G, distance]):
                return None

            return {
                "log_population": np.log(P + 1e-6),
                "log_gdp": np.log(G + 1e-6),
                "log_distance": np.log(distance + 1e-6),
                "poverty_rate": PR / 100.0,
                "elderly_persons": EP / 100.0,
                "any_insurance": AI / 100.0,
                "NHIF_cover": NHF / 100.0,
                "positive_test_rates": PTR / 100.0,
                "population_vaccinated": PV / 100.0,
            }

        except KeyError:
            return None
