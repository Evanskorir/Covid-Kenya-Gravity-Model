import geopandas as gpd
import os
import pandas as pd


class DataLoader:
    def __init__(self):
        # Paths
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self._county_data_file = os.path.join(base_dir, "../../data", "county_data.xls")
        self._geojson_file = os.path.join(base_dir, "../../data", "kenya_counties.geojson")

        self._national_confirmed_file = os.path.join(base_dir, "../../data",
                                                     "time_series_cases_covid.csv")
        self._national_deaths_file = os.path.join(base_dir, "../../data",
                                                  "time_series_deaths_covid.csv")

        # Load data
        self._load_county_data()
        self._load_national_time_series()

    def _load_county_data(self):
        df = pd.read_excel(self._county_data_file)

        # Normalize column names
        df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "") for
                      col in df.columns]
        df.rename(columns=lambda x: x.replace("__", "_"), inplace=True)

        # Detect and normalize county column
        county_col = next((col for col in df.columns if col.lower() in [
            "county", "counties", "county_name"]), None)
        if not county_col:
            raise KeyError("County column not found in county Excel file.")
        df[county_col] = df[county_col].str.strip().str.title()
        df.set_index(county_col, inplace=True)

        self.data_df = df

        # Extract static data columns
        self.population = self.get_column_as_dict("Population")
        self.land_area = self.get_column_as_dict("Land Area")
        self.density = self.get_column_as_dict("Population Density")
        self.working_population = self.get_column_as_dict("Working Population")
        self.gdp = self.get_column_as_dict("GDP")
        self.poverty_rate = self.get_column_as_dict("Poverty Rate")
        self.internet_usage = self.get_column_as_dict("Used Internet")
        self.tv_access = self.get_column_as_dict("Used Television")
        self.households_tested = self.get_column_as_dict("Households Tested")
        self.households_vaccinated = self.get_column_as_dict("Households Vaccinated")
        self.pop_tested = self.get_column_as_dict("Pop Tested")
        self.pop_vaccinated = self.get_column_as_dict("Pop Vaccinated")
        self.deaths = self.get_column_as_dict("Deaths")
        self.positive_tested_rates = self.get_column_as_dict("Number Tested Positive")
        self.number_households = self.get_column_as_dict("Number of Households")
        self.any_health_insurance = self.get_column_as_dict("Any Health Insurance")
        self.nhif = self.get_column_as_dict("National Health Insurance Fund")
        self.social_assistance = self.get_column_as_dict("Receiving Social Assistance")
        self.elderly_persons = self.get_column_as_dict("Elderly Person")
        self.covid_relief = self.get_column_as_dict("COVID-19 Relief")

        # Parse cases by date
        self.case_columns = [col for col in df.columns if col.lower().startswith("cases")]
        self.cases_by_date = {
            self._extract_date_label(col): df[col].to_dict()
            for col in self.case_columns
        }

    def _load_national_time_series(self):
        confirmed_df = pd.read_csv(self._national_confirmed_file).fillna(0)
        deaths_df = pd.read_csv(self._national_deaths_file).fillna(0)

        kenya_confirmed = confirmed_df[
                              confirmed_df['Country/Region'] == "Kenya"].iloc[:, 4:].sum()
        kenya_deaths = deaths_df[deaths_df['Country/Region'] == "Kenya"].iloc[:, 4:].sum()

        kenya_confirmed.index = pd.to_datetime(kenya_confirmed.index)
        kenya_deaths.index = pd.to_datetime(kenya_deaths.index)

        self.kenya_confirmed_series = kenya_confirmed[kenya_confirmed > 0]
        self.kenya_deaths_series = kenya_deaths[kenya_deaths > 0]

        if not self.kenya_confirmed_series.empty:
            self.first_case_date = self.kenya_confirmed_series.index[0]
            self.initial_cases = self.kenya_confirmed_series.iloc[0]
        else:
            self.first_case_date = None
            self.initial_cases = 0

    def _extract_date_label(self, col_name):
        return col_name.replace("Cases_", "").replace("_", "-")

    def get_all_data(self):
        return self.data_df

    def get_all_counties(self):
        return list(self.data_df.index)

    def get_column_as_dict(self, column_name):
        col = column_name.strip().replace(" ", "_")
        return self.data_df[col].to_dict() if col in self.data_df.columns else {}

    def get_value_for_county(self, county, column_name):
        county = county.strip().title()
        column = column_name.strip().replace(" ", "_")
        return self.data_df.at[county, column] if county in self.data_df.index else None

    def get_cases_by_date(self, date_label):
        return self.cases_by_date.get(date_label)

    def get_cases_time_series(self, county):
        county = county.strip().title()
        return {date: self.cases_by_date[date].get(county, 0) for date in self.cases_by_date}

    def load_county_shapefile(self):
        if not os.path.exists(self._geojson_file):
            raise FileNotFoundError(f"GeoJSON file not found at: {self._geojson_file}")

        gdf = gpd.read_file(self._geojson_file)
        if "NAME_1" in gdf.columns:
            gdf["NAME"] = gdf["NAME_1"].str.title()
        elif "NAME" in gdf.columns:
            gdf["NAME"] = gdf["NAME"].str.title()
        else:
            raise ValueError("County name column not found in shapefile.")

        return gdf




