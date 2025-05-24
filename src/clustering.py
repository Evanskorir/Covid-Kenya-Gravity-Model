import numpy as np
import pandas as pd
import os

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist


class CountyPeriodClustering:
    def __init__(self, cases_by_date=None, output_dir="output/clusters"):
        self.cases_by_date = cases_by_date or {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def cluster_on_structural_factors(self, variable_dicts,
                                      cluster_threshold=None,
                                      distance_metric="euclidean",
                                      show_clusters=False):
        """
        Performs hierarchical clustering on counties using selected structural variables.

        :param variable_dicts: Dict of variable name -> county-to-value dict
        :param cluster_threshold: Float value to cut the dendrogram into clusters
        :param distance_metric: Metric used to compute pairwise distance
        :param show_clusters: Print resulting clusters
        :return: (linkage_matrix, county_index)
        """

        # Build DataFrame
        df = pd.DataFrame(variable_dicts)
        df = df.dropna()
        # Standardize the data
        scaled_data = StandardScaler().fit_transform(df)

        # Compute linkage matrix
        distances = pdist(scaled_data, metric=distance_metric)
        linkage_matrix = linkage(distances, method='complete')

        return linkage_matrix, df.index

