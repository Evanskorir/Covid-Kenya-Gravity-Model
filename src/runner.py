from src.clustering import CountyPeriodClustering
from src.data_loader.coordinatesloader import CoordinateLoader
from src.data_loader.dataloader import DataLoader
from src.distance_calculator import DistanceCalculator
from src.gravity.gravity_autoregression import SpatialGravityModel
from src.gravity.gravity_cases_model import CasesGravityModel
from src.gravity.gravity_deaths_model import DeathsGravityModel
from src.plotter import Plotter


class AnalysisOrchestrator:
    def __init__(self):
        self.data = DataLoader()
        self.coords = CoordinateLoader()
        self.coordinates = self.coords.get_all_coordinates()

        self.distance_calc = DistanceCalculator(self.coordinates, method="geodesic")
        self.distance_calc.compute_all_distances()

        self.dist_from_nairobi = self.distance_calc.get_distances_from_nairobi()
        self.distance_matrix = self.distance_calc.get_all_distances()

        self.significant_variable_dicts = {
            "GDP": self.data.gdp,
            "Poverty Rate": self.data.poverty_rate,
            "Working Population": self.data.working_population,
            "TV Access": self.data.tv_access
        }

        self.plotter = Plotter(
            gravity_dict={},
            confirmed=self.data.kenya_confirmed_series,
            deaths=self.data.kenya_deaths_series
        )

    def run_all(self):
        self._run_gravity_models()
        self._run_spatial_model()
        self._generate_plots()
        self._run_clustering()

    def _run_gravity_models(self):
        print("Running Gravity Models for Cases...")
        cases_model = CasesGravityModel(
            data_loader=self.data,
            distances_from_nairobi=self.dist_from_nairobi,
            output_dir="output/cases"
        )
        for label, cases in self.data.cases_by_date.items():
            cases_model.run_model(cases, label)

        print("Running Gravity Model for Deaths...")
        deaths_model = DeathsGravityModel(
            data_loader=self.data,
            distances_from_nairobi=self.dist_from_nairobi,
            output_dir="output/deaths"
        )
        deaths_model.run_model(self.data.deaths, label="Deaths")

    def _run_spatial_model(self):
        selected_vars = {
            "CasesAug-15,-2020": ["log_gdp", "log_distance", "internet_access",
                                  "positive_test_rates", "population_vaccinated", "tv_access"],
            "CasesJuly-21,-2021": ["log_gdp", "log_distance", "log_working",
                                   "internet_access", "poverty_rate",
                                   "population_tested", "tv_access"],
            "Feb-16,-2021": ["log_gdp", "log_distance", "poverty_rate",
                             "tv_access", "internet_access", "positive_test_rates"]
        }

        model = SpatialGravityModel(
            data_loader=self.data,
            distance_matrix=self.distance_matrix,
            distances_from_nairobi=self.dist_from_nairobi,
            target_variable="cases",
            output_dir="output/spatial"
        )
        model.fit_all_models(
            selected_vars_by_label=selected_vars,
            target_dict_by_label=self.data.cases_by_date
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
            "June_2020": self.data.cases_by_date.get("June-02,-2020"),
            "Aug_2020": self.data.cases_by_date.get("CasesAug-15,-2020"),
            "Feb_2021": self.data.cases_by_date.get("Feb-16,-2021"),
            "July_2021": self.data.cases_by_date.get("CasesJuly-21,-2021")
        }

        self.plotter.plot_variable_correlation_matrix(self.data,
                                                      case_snapshots=snapshot_dict)

        self.plotter.plot_distance_heatmap(self.distance_matrix, counties)
        self.plotter.plot_gravity_vector_map_with_time_series(
            self.dist_from_nairobi, self.data.cases_by_date,
            self.coordinates, gdf, output_path="output",
            selected_dates=selected_dates
        )
        self.plotter.plot_individual()

    def _run_clustering(self):
        clustering = CountyPeriodClustering()
        linkage_matrix, labels = clustering.cluster_on_structural_factors(
            variable_dicts=self.significant_variable_dicts,
            cluster_threshold=2.0,
            show_clusters=True
        )

        snapshot = "CasesJuly-21,-2021"

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

        self.plotter.plot_combined_cluster_figure(
            shapefile_gdf=gdf,
            linkage_matrix=linkage_matrix,
            counties=labels,
            cluster_threshold=3.8,
            cases_by_date=self.data.cases_by_date,
            distance_dict=self.dist_from_nairobi,
            selected_snapshots=[snapshot],
            county_number_map=county_number_map,
            date_label=snapshot,
            save_path=f"output/combined_cluster_plot_{snapshot}.pdf"
        )

        # Choropleth & stacked bars
        self.plotter.plot_stacked_percent_bars_by_county(self.significant_variable_dicts,
                                                         output_dir="output")
        titles = {
            "Poverty Rate": "Kenya Counties by Poverty Rate (%)",
            "GDP": "Kenya Counties by GDP",
            "TV Access": "Kenya Households with TV Access (%)",
            "Working Population": "Kenya Counties by Working Population"
        }
        self.plotter.plot_multiple_choropleths(gdf, self.significant_variable_dicts, titles)
