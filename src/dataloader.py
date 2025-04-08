import os
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    def __init__(self):
        self._population_data_file = os.path.join(
            PROJECT_PATH, "../data", "county_population.xls")
        self._cases_data_file = os.path.join(
            PROJECT_PATH, "../data", "covid_cases.xls")

        self.pop_data = self._load_excel_as_dict(self._population_data_file)
        self.cases_data = self._load_excel_as_dict(self._cases_data_file)

    def _load_excel_as_dict(self, file_path):
        wb = xlrd.open_workbook(file_path)
        sheet = wb.sheet_by_index(0)
        data = {}
        for row in range(1, sheet.nrows):  # skip header
            county = str(sheet.cell_value(row, 0)).strip()
            value = sheet.cell_value(row, 1)
            data[county] = value
        wb.unload_sheet(0)
        return data

    def get_population_for_county(self, county):
        return self.pop_data.get(county)

    def get_cases_for_county(self, county):
        return self.cases_data.get(county)

    def get_all_population(self):
        return self.pop_data

    def get_all_cases(self):
        return self.cases_data

