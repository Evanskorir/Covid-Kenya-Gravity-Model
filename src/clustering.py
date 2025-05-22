import numpy as np
import pandas as pd
import os

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist


class CountyPeriodClustering:
    def __init__(self, cases_by_date, output_dir="output/clusters"):
        self.cases_by_date = cases_by_date
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_linkage_for_date(self, date_label,
                                 log_transform=False,
                                 distance_metric="euclidean",
                                 cluster_threshold=None,
                                 show_clusters=False):

        cases_dict = self.cases_by_date[date_label]
        counties = sorted(cases_dict)
        values = [[cases_dict.get(county, 0)] for county in counties]
        df = pd.DataFrame(values, index=counties, columns=["Cases"])

        if log_transform:
            df["Cases"] = df["Cases"].apply(lambda x: np.log1p(x))

        scaled_data = StandardScaler().fit_transform(df)
        distances = pdist(scaled_data, metric=distance_metric)
        linkage_matrix = linkage(distances, method='complete')

        if cluster_threshold is not None:
            clusters = fcluster(linkage_matrix, t=cluster_threshold, criterion='distance')
        else:
            clusters = np.ones(len(df), dtype=int)

        cluster_df = pd.DataFrame({"County": df.index, "Cluster": clusters})
        if show_clusters:
            print(f"Cluster assignments for {date_label}:")
            print(cluster_df.sort_values("Cluster").to_string(index=False))

        return linkage_matrix, df.index

