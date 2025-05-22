# Modeling the Spatial Spread of COVID-19 in Kenya: A Gravity-Based and Clustering Approach
## Introduction
This repository investigates the spatial spread of COVID-19 across Kenyan counties using gravity-based and 
spatial autoregressive (SAR) models. It leverages county-level data on confirmed cases, population, GDP, poverty rates,
working population, and distance from Nairobi. Cluster analysis is also performed to uncover patterns in the geographical distribution of COVID-19.

## Data Files
```
data: Socio-economic indicators and cumulative cases collected from different sources "
   ├── coordinates      
   │   ├── coordinates   # JSON files containing latitude and longitude of a county
   ├── county_data.xls   # Excel file with data such as population, GDP, poverty, working population, etc.
   ├── time series cases covid # country's time series confirmed covid-19 cases     
   ├── time series deaths covid # country's time series confirmed covid-19 deaths     
   └── kenya counties geojson  # JSON files containing latitude and longitude 
  
  ├── socio-economic indicators   # population, GDP, poverty, working population, etc.
  ├── county cumulative cases     # Cumulative cases on June 02, 2020, August 15, 2020, February 16, 2021 and July 21, 2021
  └── coordinates data      # JSON files containing latitude and longitude of a county.
```

## Folder Structure
```
data                
src                    
 ├── data_loader        
 │   ├── coordinatesloader     
 │   └── dataloader   
 ├── gravity        
 │   ├── gravity_autoregression    
 │   ├── gravity_base_model 
 │   ├── gravity_calculator   
 │   ├── gravity_cases_model 
 │   └── gravity_deaths_model   
 ├── clustering
 ├── distance_calculator
 ├── plotter               
 └── runner
main 
README
```
## File Details
#### `src/data_loader/`
- **`coordinatesloader.py`**: Loads and normalizes geographic coordinates (latitude and longitude) for Kenyan counties 
from JSON files to support spatial analysis and mapping.
- **`dataloader.py`**: Loads, preprocesses, and provides access to Kenya's COVID-19 time series, demographic, 
socioeconomic, and geospatial data for use in modeling and visualization.
#### `src/gravity/`
- **`gravity_autoregression.py`**: Implements spatial regression to model COVID-19 outcomes by incorporating 
spatial lag effects based on inter-county distances and socioeconomic variables.
- **`gravity_base_model.py`**: Provides a framework for gravity-based regression modeling, including feature extraction, 
data preparation, and automated variable selection using backward elimination for spatial epidemiological analysis.
- **`gravity_calculator.py`**: Computes a gravity matrix estimating interaction strength between counties based on 
case counts, population, and distances.
- **`gravity_cases_model.py`**: A gravity-based regression model that predicts COVID-19 case spread using socioeconomic,
demographic, and distance-related features.
- **`gravity_deaths_model.py`**:  A gravity-based model for analyzing COVID-19 deaths using distance, population, 
healthcare access, and vulnerability indicators.
#### `src/`
- **`clustering.py`**:  Performs hierarchical clustering of counties based on COVID-19 case data for a specific date, 
with optional log transformation and visualization.
- **`distance_calculator.py`**: Computes pairwise distances between counties using either geodesic or 
haversine methods for spatial analysis.
- **`plotter.py`**: Generates a wide range of visualizations—including heatmaps, time-series maps, dendrograms, and 
choropleths—to analyze spatial and temporal COVID-19 dynamics across Kenyan counties.
- **`runner.py`**: Coordinates the entire analysis pipeline—data loading, gravity modeling, spatial regression, 
visualization, and clustering—for COVID-19 patterns across Kenyan counties.
- **`main.py`**: Runs the full COVID-19 spatial analysis pipeline by initializing and executing the AnalysisOrchestrator

## Implementation
To run the spatial analysis, follow these steps:
1. Open `main.py` 
2. Run the analysis with these steps:` 
#### Initialize the AnalysisOrchestrator
```  analysis = AnalysisOrchestrator() ```
#### generate all the necessary plots
```analysis.run_all() ```

## Output
```
output/cases
     ├── ols_full_Cases(Date; June-02-2020, Aug-15-2020, Feb-16-2021, July-21-2021)
     ├── ols_selected_Cases(Date; June-02-2020, Aug-15-2020, Feb-16-2021, July-21-2021)
     ├── spatial
     │    └── sar_summary(Date; Aug-15-2020, Feb-16-2021, July-21-2021)
     ├──   combined_cluster_plot_Cases(Date).pdf 
     ├──   correlation_matrix.pdf
     ├──    county_stacked_percent.pdf    
     ├──   distance_matrix_heatmap.pdf
     ├──  kenya_gravity_map_timeseries.png
```

## Requirement
This project is developed and tested with Python 3.8 or higher. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
