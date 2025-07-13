from src.clustering import CountyPeriodClustering
from src.data_loader.coordinatesloader import CoordinateLoader
from src.data_loader.dataloader import DataLoader
from src.distance_calculator import DistanceCalculator
from src.gravity.gravity_autoregression import SpatialGravityModel
from src.gravity.gravity_cases_model import CasesGravityModel
from src.gravity.gravity_deaths_model import DeathsGravityModel
from src.plotter import Plotter


class AnalysisOrchestrator:
    def __init__(self, hub="Nairobi", use_prevalence=True):
        self.data = DataLoader()
        self.coords = CoordinateLoader()
        self.coordinates = self.coords.get_all_coordinates()

        self.hub = hub
        self.use_prevalence = use_prevalence

        self.distance_calc = DistanceCalculator(self.coordinates, method="geodesic")
        self.distance_calc.compute_all_distances()

        self.dist_from_county_hub = self.distance_calc.get_distances_from_hub(hub_name=hub)
        self.distance_matrix = self.distance_calc.get_all_distances()

        self.significant_variable_dicts = {
            "GDP": self.data.gdp,
            "Poverty Rate": self.data.poverty_rate,
            "Working Population": self.data.working_population,
            "population_tested": self.data.pop_tested,
            "Number of Households": self.data.number_households,
            "TV Access": self.data.tv_access
        }

        self.plotter = Plotter(
            gravity_dict={},
            confirmed=self.data.kenya_confirmed_series,
            deaths=self.data.kenya_deaths_series
        )

        self.target_data_by_date = (
            self.data.compute_prevalence_by_date() if use_prevalence else
            self.data.cases_by_date
        )
        self.target_var_name = "prevalence" if self.use_prevalence else "cases"

    def run_all(self):
        self._run_gravity_models()
        self._run_spatial_model()
        self._generate_plots()
        self._run_clustering()

    def _run_gravity_models(self):
        print("Running Gravity Models for Cases...")
        cases_model = CasesGravityModel(
            data_loader=self.data,
            distances_from_county_hub=self.dist_from_county_hub,
            output_dir="output/cases",
            target_variable=self.target_var_name
        )

        for label, cases in self.target_data_by_date.items():
            cases_model.run_model(cases, label)

        print("Running Gravity Model for Deaths...")
        deaths_model = DeathsGravityModel(
            data_loader=self.data,
            distances_from_county_hub=self.dist_from_county_hub,
            output_dir="output/deaths"
        )
        deaths_model.run_model(self.data.deaths, label="Deaths")

    def _run_spatial_model(self):
        selected_vars = {
            "CasesAug-15,-2020": ["log_gdp", "log_distance", "log_working",
                                    "poverty_rate", "log_house_holds", "internet_access",
                                    "population_tested", "tv_access"],
            "CasesJuly-21,-2021": ["log_gdp", "log_distance", "log_working",
                                    "poverty_rate", "log_house_holds", "internet_access",
                                    "population_tested", "tv_access"],
            "Feb-16,-2021": ["log_gdp", "log_distance", "log_working",
                                    "poverty_rate", "log_house_holds", "internet_access",
                                    "population_tested", "tv_access"]
        }

        model = SpatialGravityModel(
            data_loader=self.data,
            distance_matrix=self.distance_matrix,
            distances_from_county_hub=self.dist_from_county_hub,
            target_variable=self.target_var_name,
            output_dir="output/spatial"
        )
        model.fit_all_models(
            selected_vars_by_label=selected_vars,
            target_dict_by_label=self.target_data_by_date
        )

    def _generate_plots(self):
        counties = self.data.get_all_counties()
        gdf = self.data.load_county_shapefile()
        selected_dates = [
            "June-02,-2020",
            "CasesAug-15,-2020",
            "Feb-16,-2021",
            "CasesJuly-21,-2021"
        ]
        snapshot_dict = {
            "June_2020": self.target_data_by_date.get("June-02,-2020"),
            "Aug_2020": self.target_data_by_date.get("CasesAug-15,-2020"),
            "Feb_2021": self.target_data_by_date.get("Feb-16,-2021"),
            "July_2021": self.target_data_by_date.get("CasesJuly-21,-2021")
        }

        label_type = "Prevalence" if self.use_prevalence else "Cases"
        self.plotter.plot_variable_correlation_matrix(
            self.data,
            case_snapshots=snapshot_dict,
            output_path="output/correlation_matrix.pdf",
            label_type=label_type
        )

        self.plotter.plot_distance_heatmap(self.distance_matrix, counties)

        self.plotter.plot_gravity_vector_map_with_time_series(
            self.dist_from_county_hub,
            self.target_data_by_date,
            self.coordinates,
            gdf,
            output_path="output",
            selected_dates=selected_dates,
            hub_name=self.hub
        )

        self.plotter.plot_individual()

    def _run_clustering(self):
        clustering = CountyPeriodClustering(cases_by_date=self.target_data_by_date)
        snapshot = "CasesJuly-21,-2021"

        # --- Map ---
        ordered_counties = [
            "mombasa", "kwale", "kilifi", "tana river", "lamu", "taita taveta", "garissa", "wajir",
            "mandera", "marsabit", "isiolo", "meru", "tharaka nithi", "embu", "kitui", "makueni",
            "machakos", "nyandarua", "nyeri", "kirinyaga", "muranga", "kiambu", "turkana",
            "west pokot", "samburu", "trans nzoia", "uasin gishu", "elgeyo marakwet", "nandi",
            "baringo", "laikipia", "nakuru", "narok", "kajiado", "kericho", "bomet", "kakamega",
            "vihiga", "bungoma", "busia", "siaya", "kisumu", "homa bay", "migori", "kisii",
            "nyamira", "nairobi"
        ]
        county_number_map = {c: i + 1 for i, c in enumerate(ordered_counties)}
        gdf = self.data.load_county_shapefile()

        # === Cluster depending on data type ===
        if self.use_prevalence:
            linkage_matrix, county_labels = clustering.cluster_on_prevalence_snapshot(
                snapshot_label=snapshot,
                cluster_threshold=1.0,
                show_clusters=True
            )
            cluster_threshold = 1.0
            label_type = "prevalence"
        else:
            linkage_matrix, county_labels = clustering.cluster_on_structural_factors(
                variable_dicts=self.significant_variable_dicts,
                cluster_threshold=5.7,
                show_clusters=True
            )
            cluster_threshold = 5.7
            label_type = "cases"

        # === Plot combined view ===
        self.plotter.plot_combined_cluster_view(
            shapefile_gdf=gdf,
            linkage_matrix=linkage_matrix,
            counties=county_labels,
            cluster_threshold=cluster_threshold,
            cases_by_date=self.target_data_by_date,
            distance_dict=self.dist_from_county_hub,
            selected_snapshots=[snapshot],
            county_number_map=county_number_map,
            date_label=f"{snapshot}_{label_type}",
            save_path=f"output/cluster_{label_type}.pdf",
            label_type=label_type
        )

        # === Optional: additional plots ===
        self.plotter.plot_stacked_percent_bars_by_county(
            self.significant_variable_dicts,
            output_dir="output"
        )

        titles = {
            "Poverty Rate": "Kenya Counties by Poverty Rate (%)",
            "GDP": "Kenya Counties by GDP",
            "TV Access": "Kenya Households with TV Access (%)",
            "Working Population": "Kenya Counties by Working Population"
        }

        significant_variable_dicts = {
            "GDP": self.data.gdp,
            "Poverty Rate": self.data.poverty_rate,
            "Working Population": self.data.working_population,
            "TV Access": self.data.tv_access
        }

        self.plotter.plot_multiple_choropleths(gdf, significant_variable_dicts, titles)

