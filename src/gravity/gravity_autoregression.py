import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm

from matplotlib.backends.backend_pdf import PdfPages


class SpatialGravityModel:
    def __init__(self, data_loader, distance_matrix, distances_from_nairobi,
                 alpha=0.05, output_dir="output", target_variable="cases"):
        self.data = data_loader
        self.distance_matrix = distance_matrix
        self.distances = distances_from_nairobi
        self.alpha = alpha
        self.output_dir = output_dir
        self.target_variable = target_variable
        os.makedirs(output_dir, exist_ok=True)

    def _build_spatial_lag(self, log_target_series):
        counties = log_target_series.index.tolist()
        W = pd.DataFrame(0.0, index=counties, columns=counties)

        for i in counties:
            weights = {}
            total_weight = 0.0
            for j in counties:
                if i == j:
                    continue
                d = self.distance_matrix.get((i, j)) or self.distance_matrix.get((j, i))
                if d and d > 0:
                    weight = 1 / d
                    weights[j] = weight
                    total_weight += weight

            for j, w in weights.items():
                W.loc[i, j] = w / total_weight if total_weight > 0 else 0.0

        spatial_lag = W.values @ log_target_series.values
        return pd.Series(spatial_lag, index=counties, name="spatial_lag")

    def _save_summary_as_pdf(self, model, filename):
        summary_text = model.summary().as_text()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0, 1, summary_text, fontsize=9, fontfamily='monospace', va='top')
        path = os.path.join(self.output_dir, filename)
        with PdfPages(path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        print(f"Saved SAR summary to: {path}")

    def fit_model_for_single_point(self, label, selected_vars, target_dict):
        print(f"Fitting spatial gravity model for {label}...")

        records = []

        for county, distance in self.distances.items():
            if county == "Nairobi":
                continue

            N = target_dict.get(county)
            if N is None:
                continue

            try:
                record = {"county": county, f"log_{self.target_variable}": np.log(N)}

                for var in selected_vars:
                    if var == "log_population":
                        val = self.data.population[county]
                        record[var] = np.log(val)
                    elif var == "log_gdp":
                        val = self.data.gdp[county]
                        record[var] = np.log(val)
                    elif var == "log_working":
                        val = self.data.working_population[county]
                        record[var] = np.log(val)
                    elif var == "log_distance":
                        record[var] = np.log(distance)
                    elif var == "poverty_rate":
                        record[var] = self.data.poverty_rate[county] / 100.0
                    elif var == "internet_access":
                        record[var] = self.data.internet_usage[county] / 100.0
                    elif var == "tv_access":
                        record[var] = self.data.tv_access[county] / 100.0
                    elif var == "population_tested":
                        record[var] = self.data.pop_tested[county] / 100.0
                    elif var == "positive_test_rates":
                        record[var] = self.data.positive_tested_rates[county] / 100.0
                    elif var == "population_vaccinated":
                        record[var] = self.data.pop_vaccinated[county] / 100.0
                records.append(record)

            except Exception:
                continue

        df = pd.DataFrame(records).set_index("county")

        if df.empty:
            print(f"Skipping {label}: no valid data.")
            return

        log_col = f"log_{self.target_variable}"
        df["spatial_lag"] = self._build_spatial_lag(df[log_col])

        X = df[selected_vars + ["spatial_lag"]]
        X = sm.add_constant(X)
        y = df[log_col]

        model = sm.OLS(y, X).fit()
        self._save_summary_as_pdf(model, f"sar_summary_{label}.pdf")

    def fit_all_models(self, selected_vars_by_label, target_dict_by_label):
        for label, selected_vars in selected_vars_by_label.items():
            target_dict = target_dict_by_label.get(label)
            if target_dict:
                self.fit_model_for_single_point(label, selected_vars, target_dict)
            else:
                print(f"Missing data for {label}")
