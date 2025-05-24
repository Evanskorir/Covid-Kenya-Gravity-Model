import matplotlib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from matplotlib import gridspec, patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, fcluster
matplotlib.use("Agg")


class Plotter:
    def __init__(self, gravity_dict, confirmed, deaths,
                 output_dir="output"):
        self.dendo_colors = None
        self.gravity_dict = gravity_dict
        self.output_dir = output_dir
        self.gravity_df = self._to_dataframe()

        # Daily data (difference from cumulative data)
        self.confirmed = confirmed.diff().fillna(0)
        self.deaths = deaths.diff().fillna(0)

        # Cumulative data (as is)
        self.cumulative_confirmed = confirmed
        self.cumulative_deaths = deaths

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Region Definitions ---
        self.region_order = {
            "Nairobi": ["Nairobi"],
            "Coast": ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita Taveta"],
            "North Eastern": ["Garissa", "Wajir", "Mandera"],
            "Eastern": ["Kitui", "Machakos", "Makueni", "Meru", "Embu", "Tharaka Nithi",
                "Isiolo", "Marsabit"],
            "Central": ["Muranga", "Kiambu", "Nyandarua", "Nyeri", "Kirinyaga"],
            "Rift Valley": ["Baringo", "Bomet", "Elgeyo Marakwet", "Kajiado", "Kericho",
                "Laikipia", "Nakuru", "Nandi", "Narok", "Samburu", "Trans Nzoia",
                "Turkana", "Uasin Gishu", "West Pokot"],
            "Western": ["Bungoma", "Busia", "Kakamega", "Vihiga"],
            "Nyanza": ["Homa Bay", "Kisii", "Kisumu", "Migori", "Nyamira", "Siaya"]
        }
        self.correction_map = {
            "Homabay": "Homa Bay", "Murang'A": "Muranga", "Taitataveta": "Taita Taveta",
            "Tanariver": "Tana River", "Transnzoia": "Trans Nzoia",
            "Uasingishu": "Uasin Gishu", "Westpokot": "West Pokot"
        }

    @staticmethod
    def normalize_county_name(name):
        name = name.lower().replace("-", " ").replace("_", " ")
        name = name.replace("’", "'").replace("", "'").replace("'", "")
        name = " ".join(word.capitalize() for word in name.split())
        return name

    def _to_dataframe(self):
        # Normalize all county names in the keys
        normalized_gravity = {
            (self.normalize_county_name(i), self.normalize_county_name(j)): val
            for (i, j), val in self.gravity_dict.items()
        }

        counties = sorted(set([i for i, _ in normalized_gravity.keys()] +
                              [j for _, j in normalized_gravity.keys()]))

        df = pd.DataFrame(index=counties, columns=counties)
        for (origin, dest), value in normalized_gravity.items():
            df.loc[origin, dest] = value
        return df.astype(float)

    @staticmethod
    def convert_distance_dict_to_df(distance_dict, counties):
        """
        Converts a pairwise distance dictionary to a square DataFrame.
        """
        df = pd.DataFrame(index=counties, columns=counties)

        for (origin, dest), val in distance_dict.items():
            df.at[origin, dest] = val

        # Fill diagonals with 0 and convert to float
        np.fill_diagonal(df.values, 0)
        return df.astype(float)

    def plot_distance_heatmap(self, distance_dict, counties,
                              filename="distance_matrix_heatmap.pdf"):
        """
        Plots a square heatmap of county-to-county distances using Matplotlib with annotations.
        """
        distance_df = self.convert_distance_dict_to_df(distance_dict, counties)
        values = distance_df.values
        values = distance_df.values[::-1]

        fig, ax = plt.subplots(figsize=(20, 20))

        # Heatmap with colormap
        norm = Normalize(vmin=0, vmax=600)
        im = ax.imshow(values, cmap="coolwarm", norm=norm)

        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(counties)))
        ax.set_yticks(np.arange(len(counties)))
        ax.set_xticklabels(counties, rotation=90, fontsize=18)
        ax.set_yticklabels(counties[::-1], fontsize=18)

        ax.set_xlabel("County of origin", fontsize=25, labelpad=15)
        ax.set_ylabel("Destination County", fontsize=25, labelpad=15)

        ax.tick_params(which="minor", bottom=False, left=False)

        # Axis border visibility
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

        # Colorbar setup
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.4)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Distance (km)", fontsize=24)
        cb.set_ticks([100, 200, 300, 400, 500, 600])
        cb.ax.tick_params(labelsize=25)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"Distance matrix heatmap saved to {save_path}")

    @staticmethod
    def plot_variable_correlation_matrix(data, case_snapshots=None,
                                         output_path="output/correlation_matrix.pdf"):

        df = data.get_all_data()

        # Define core socioeconomic variables
        base_cols = [
            "Population", "Land Area", "Population Density",
            "Working Population", "GDP", "Poverty Rate",
            "Used Internet", "Used Television", "Pop Tested", "Pop Vaccinated",
            "Number Tested Positive", "Number of Households",
            "Households Tested", "Households Vaccinated"
        ]

        col_map = {col.replace(" ", "_"): col for col in base_cols}

        # Add case columns to DataFrame and build list for reordering
        case_cols = []
        if case_snapshots:
            for label, series in case_snapshots.items():
                case_col = f"Cases_{label}"
                df[case_col] = series
                case_cols.append(case_col)

        # Final column order: cases first, then socioeconomic
        final_cols = case_cols + [col.replace(" ", "_") for col in base_cols]
        df = df[final_cols].dropna()
        corr = df.corr().round(2)
        corr = corr.iloc[::-1]

        # Define axis labels
        labels = [col_map.get(col, col.replace("_", " ")) for col in corr.columns]
        fig, ax = plt.subplots(figsize=(15, 12))
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        # Ticks
        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=18)
        ax.set_yticklabels(labels[::-1], fontsize=18)

        ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

        # Annotate each cell
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha='center', va='center',
                        fontsize=11, color=text_color, fontweight="bold")

        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"correlation matrix saved to {output_path}")

    @staticmethod
    def plot_gravity_vector_map_with_time_series(distances_from_nairobi, cases_by_date,
                                                 coordinates_dict, shapefile_gdf, output_path,
                                                 selected_dates=None):
        if selected_dates is None:
            selected_dates = list(cases_by_date.keys())[:5]

        nairobi_coord = coordinates_dict["Nairobi"]
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        shapefile_gdf.plot(ax=ax, edgecolor="gray", facecolor="white", linewidth=0.7)
        date_colors = {
            date: color for date, color in zip(
                selected_dates,
                ["red", "#984ea3", "#33a02c", "#ff7f00", "#e31a1c"]
            )
        }
        distance_colors = {
            "≤100 km": "black",
            "100–200 km": "blue",
            "200–300 km": "cyan",
            "300–400 km": "orange",
            ">400 km": "red"
        }

        def get_band_and_color(distance):
            if distance <= 100:
                return "≤100 km", distance_colors["≤100 km"]
            elif distance < 200:
                return "100–200 km", distance_colors["100–200 km"]
            elif distance < 300:
                return "200–300 km", distance_colors["200–300 km"]
            elif distance < 400:
                return "300–400 km", distance_colors["300–400 km"]
            else:
                return ">400 km", distance_colors[">400 km"]

        used_bands = set()
        for county, dist in distances_from_nairobi.items():
            if county not in coordinates_dict or county == "Nairobi":
                continue

            x1, y1 = nairobi_coord[1], nairobi_coord[0]
            x2, y2 = coordinates_dict[county][1], coordinates_dict[county][0]
            band_label, color = get_band_and_color(dist)
            used_bands.add(band_label)

            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-|>', color=color,
                linewidth=1.2, alpha=0.7,
                connectionstyle="arc3,rad=0.1", capstyle='round'
            ))

            total_cases = sum(cases_by_date[date].get(county, 0) for date in selected_dates)
            if total_cases <= 0:
                continue

            num_dates = len(selected_dates)
            radius_offset = 0.10

            for i, date in enumerate(selected_dates):
                case_count = cases_by_date[date].get(county, 0)
                if case_count <= 0:
                    continue

                angle = (2 * np.pi / num_dates) * i
                dx = radius_offset * np.cos(angle)
                dy = radius_offset * np.sin(angle)
                x_offset, y_offset = x2 + dx, y2 + dy
                radius = 0.1 * np.sqrt(case_count) / 100

                ax.add_patch(Circle(
                    (x_offset, y_offset), radius=radius,
                    color=date_colors[date], alpha=0.7,
                    ec='black', lw=0.3, zorder=5 + i
                ))

            text = ax.text(x2, y2, county, fontsize=9,
                           ha='left', va='bottom', color='black')
            text.set_path_effects([
                patheffects.Stroke(linewidth=2.5, foreground='white'),
                patheffects.Normal()
            ])

        ax.plot(nairobi_coord[1], nairobi_coord[0], marker="*", color="black",
                markersize=20, zorder=10)
        ax.text(nairobi_coord[1], nairobi_coord[0], "Nairobi",
                fontsize=11, ha='right', va='top', color='black', fontweight='bold')

        # --- DISTANCE LEGEND ---
        line_legend = [
            Line2D([], [], color=distance_colors[band], linewidth=2, label=band)
            for band in distance_colors if band in used_bands
        ]

        legend_distance = ax.legend(
            handles=[Line2D([], [], linestyle="None", label="Distance Bands")] + line_legend,
            loc='lower left',
            bbox_to_anchor=(0.01, 0.15),
            fontsize=12,
            frameon=True,
            edgecolor="gray",
            facecolor="white",
            borderpad=1,
            handletextpad=1.5
        )
        ax.add_artist(legend_distance)

        # --- CASE COUNT LEGEND ---
        case_bins = ["≤100", "100–500", "500–1000", "1000–2000", "2000–5000", ">5000"]
        case_values = [50, 300, 750, 1500, 3500, 6000]
        marker_sizes = [2 * (0.1 * np.sqrt(v) / 100) * 100 for v in case_values]

        case_legend = [
            Line2D([], [], marker='o', linestyle='None', color='none',
                   markerfacecolor='white', markeredgecolor='black',
                   markeredgewidth=0.5, markersize=size,
                   label=label, alpha=0.6)
            for size, label in zip(marker_sizes, case_bins)
        ]

        # --- DATE LEGEND ---
        date_label_map = {
            selected_dates[0]: "June 02, 2020",
            selected_dates[1]: "August 15, 2020",
            selected_dates[2]: "February 16, 2021",
            selected_dates[3]: "July 21, 2021",
        }
        for date in selected_dates:
            if date not in date_label_map:
                date_label_map[date] = str(date)

        date_legend = [
            Line2D([0], [0], marker='o', linestyle='None', color='none',
                   markerfacecolor=date_colors[date], markeredgecolor='black',
                   markersize=10, alpha=0.7, label=date_label_map[date])
            for date in selected_dates
        ]

        # --- COMBINED CASE + DATE LEGEND ---
        legend_combined = ax.legend(
            handles=[
                Line2D([], [], linestyle="None", label="Case Counts"),
                *case_legend,
                Line2D([], [], linestyle="None", label=""),
                Line2D([], [], linestyle="None", label="Reporting Dates"),
                *date_legend
            ],
            loc='lower left',
            bbox_to_anchor=(0.01, 0.01),
            fontsize=12,
            ncol=2,
            columnspacing=2.0,
            handletextpad=1.5,
            frameon=True,
            edgecolor="gray",
            facecolor="white",
            borderpad=1
        )
        ax.add_artist(legend_combined)

        # Map border
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               linewidth=2.5, edgecolor='black',
                               facecolor='none', zorder=15))

        ax.axis("off")
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, "kenya_gravity_map_timeseries.png")
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"Time-series gravity map saved to {save_path}")

    def export_to_excel(self, filename="gravity_matrix.xlsx"):
        save_path = os.path.join(self.output_dir, filename)
        self.gravity_df.to_excel(save_path)
        print(f" Gravity matrix exported to {save_path}")

    @staticmethod
    def _plot_and_save(data, label, color, filename, cumulative=False):
        """Plot and save daily or cumulative dataset from Johns Hopkins."""
        plt.figure(figsize=(12, 6))

        # Apply a 7-day rolling average for a smooth curve
        smoothed_data = data.rolling(7, center=True).mean()

        # Plot the smoothed data without fill
        plt.plot(smoothed_data, label=f'{label}', color=color, linewidth=2, alpha=0.9)

        # Different title for cumulative data
        plt.ylabel(f'Number of {label}', fontsize=14, color='#444444', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.xticks(rotation=30, color='#444444', fontweight='bold')
        plt.yticks(color='#444444', fontweight='bold')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

        # Remove figure borders
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.tight_layout()

        # Save the plot
        plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        plot_file_path = os.path.join(plot_dir, filename)
        plt.savefig(plot_file_path, format='pdf', transparent=True)
        plt.close()

    def plot_individual(self):
        """Generate and save plots for daily and cumulative COVID-19 data."""
        self._plot_and_save(self.confirmed, 'Confirmed Cases', 'blue',
                            'confirmed_cases_daily.pdf')
        self._plot_and_save(self.deaths, 'Deaths', 'red', 'deaths_daily.pdf')

        # Plot cumulative individual cases
        self._plot_and_save(self.cumulative_confirmed, 'Cumulative Confirmed Cases', 'blue',
                            'confirmed_cases_cumulative.pdf', cumulative=True)
        self._plot_and_save(self.cumulative_deaths, 'Cumulative Deaths', 'red',
                            'deaths_cumulative.pdf', cumulative=True)

    @staticmethod
    def normalize_county_name(name):
        return name.strip().lower()

    @staticmethod
    def get_color_map():
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return {f"C{i}": color_cycle[i % len(color_cycle)] for i in range(10)}

    def plot_dendrogram(self, linkage_matrix, counties, county_number_map, ax=None,
                        date_label=None, threshold=None, tight=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

        numeric_labels = [
            str(county_number_map.get(self.normalize_county_name(c), "?"))
            for c in counties
        ]
        dendro = dendrogram(
            linkage_matrix,
            labels=numeric_labels,
            leaf_rotation=90,
            leaf_font_size=16,
            ax=ax,
            color_threshold=threshold
        )

        color_map = self.get_color_map()
        raw_colors = dict(zip(dendro['ivl'], dendro['leaves_color_list']))
        self.dendo_colors = {label: color_map.get(code,
                                                  code) for label, code in raw_colors.items()}

        for lbl in ax.get_xmajorticklabels():
            label = lbl.get_text()
            lbl.set_color(self.dendo_colors.get(label, "black"))
            lbl.set_fontweight("medium")

        for line in ax.get_lines():
            if max(line.get_ydata()) > 0:
                line.set_color("darkblue")
                line.set_linewidth(1.0)

        ax.set_ylabel("Cluster distance", fontsize=25, fontweight="bold", color="darkblue")
        ax.tick_params(axis='y', labelsize=20, width=4.5, length=10, colors="darkblue")
        ax.tick_params(axis='x', bottom=False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color("darkblue")
        ax.spines['left'].set_linewidth(2)

        for tick in ax.get_xticklines():
            tick.set_visible(False)

        if tight:
            plt.tight_layout()

    def plot_cluster_map_from_dendrogram(self, shapefile_gdf,
                                         linkage_matrix,
                                         counties,
                                         county_number_map,
                                         cluster_threshold,
                                         ax=None):
        # --- Normalize function ---
        def normalize_name(name):
            return name.lower().replace("-", " ").replace("_", " ").strip()

        # --- Apply normalization ---
        counties = [normalize_name(c) for c in counties]
        shapefile_gdf = shapefile_gdf.rename(columns={"NAME": "County"})
        shapefile_gdf["County"] = shapefile_gdf["County"].apply(normalize_name)

        correction_map = {
            "homabay": "homa bay",
            "murang'a": "muranga",
            "taitataveta": "taita taveta",
            "tanariver": "tana river",
            "transnzoia": "trans nzoia",
            "uasingishu": "uasin gishu",
            "westpokot": "west pokot"
        }
        shapefile_gdf["County"] = shapefile_gdf["County"].apply(
            lambda name: correction_map.get(name, name)
        )
        # --- Assign cluster labels ---
        cluster_labels = fcluster(linkage_matrix,
                                  t=cluster_threshold, criterion='distance')
        # --- Normalize county_number_map ---
        normalized_county_number_map = {
            normalize_name(k): v for k, v in county_number_map.items()
        }
        # --- Create cluster assignment DataFrame ---
        cluster_df = pd.DataFrame({
            "County": counties,
            "Cluster": cluster_labels,
        })

        cluster_df["Number"] = cluster_df["County"].map(normalized_county_number_map)
        cluster_df["NumberStr"] = cluster_df["Number"].astype(int).astype(str)
        cluster_df["Color"] = cluster_df["NumberStr"].map(self.dendo_colors)

        # --- Merge with shapefile ---
        merged = shapefile_gdf.merge(cluster_df, how="left", on="County")

        # --- Plot ---
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 14))

        merged.plot(ax=ax,
                    facecolor=merged["Color"].fillna("lightgrey"),
                    edgecolor="black",
                    linewidth=0.5)
        # --- Annotate with county numbers ---
        for _, row in merged.iterrows():
            if row.geometry and pd.notna(row["Number"]):
                centroid = row.geometry.centroid
                label = str(int(row["Number"]))
                color = self.dendo_colors.get(label, "black")

                text_color = "black"

                txt = ax.text(centroid.x, centroid.y, label,
                              fontsize=14.5, ha="center", va="center",
                              weight="extra bold", color=text_color, zorder=10)
                txt.set_path_effects([
                    patheffects.Stroke(linewidth=3.0, foreground='white'),
                    patheffects.Normal()
                ])

        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_aspect("auto")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.axis("off")

    def plot_avg_cases_vs_distance_scatter(self, cases_by_date, distance_dict,
                                           linkage_matrix, counties,
                                           county_number_map,
                                           cluster_threshold,
                                           selected_snapshots,
                                           ax=None):
        counties = [self.normalize_county_name(c) for c in counties]
        distance_dict = {self.normalize_county_name(k): v for k, v in distance_dict.items()}
        cases_by_date = {
            snap: {self.normalize_county_name(k): v for k, v in cases.items()}
            for snap, cases in cases_by_date.items()
        }

        target_date = selected_snapshots[0]
        case_dict = cases_by_date.get(target_date)
        nairobi = self.normalize_county_name("Nairobi")
        distance_dict[nairobi] = 0
        if nairobi not in counties:
            counties.append(nairobi)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        for county in counties:
            num = county_number_map.get(county)
            if num is None:
                continue

            label = str(num)
            color = self.dendo_colors.get(label, "gray")
            x = case_dict.get(county)
            y = distance_dict.get(county)

            if x is not None and y is not None:
                ax.scatter(x, y, color=color, edgecolor='black', linewidth=0.3,
                           s=55, zorder=3)
                ax.text(x + 60, y + 5, label,
                        fontsize=15, ha="left", va="bottom", weight="bold",
                        color=color, zorder=5,
                        path_effects=[
                            patheffects.Stroke(linewidth=1.8, foreground='white'),
                            patheffects.Normal()
                        ])

        ax.set_xlabel(f"Confirmed COVID-19 Cases as at July 21, 2021", fontsize=22,
                      fontweight="bold")
        ax.set_ylabel("Distance from Nairobi (km)", fontsize=22, fontweight="bold")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(2.5)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_position(("outward", 10))

        ax.minorticks_off()
        ax.tick_params(axis='both', which='major', direction='out',
                       length=8, width=3, colors='black',
                       bottom=True, top=False, left=True, right=False, labelsize=15)

    def plot_combined_cluster_figure(self, shapefile_gdf, linkage_matrix, counties,
                                     cluster_threshold, cases_by_date, distance_dict,
                                     selected_snapshots, county_number_map,
                                     date_label, save_path):

        fig = plt.figure(figsize=(25, 22))
        gs = gridspec.GridSpec(3, 6, height_ratios=[0.8, 2.5, 2.2],
                               width_ratios=[1.3, 1, 1, 1, 1, 1])
        gs.update(hspace=0.4, wspace=0.2)

        ax_abbrev = fig.add_subplot(gs[0, :])
        ax_dendro = fig.add_subplot(gs[1, 0:4])
        ax_map = fig.add_subplot(gs[1:3, 4:6])
        ax_scatter = fig.add_subplot(gs[2, 0:4])

        # Adjust map axis position
        box = ax_map.get_position()
        ax_map.set_position([
            box.x0 - 0.02,
            box.y0 - 0.09,
            box.width * 1.5,
            box.height * 1.3
        ])

        # Plot dendrogram
        self.plot_dendrogram(
            linkage_matrix, counties, county_number_map, ax=ax_dendro,
            date_label=date_label, threshold=cluster_threshold, tight=False
        )

        # Plot cluster map
        self.plot_cluster_map_from_dendrogram(
            shapefile_gdf=shapefile_gdf,
            linkage_matrix=linkage_matrix,
            counties=counties,
            county_number_map=county_number_map,
            cluster_threshold=cluster_threshold,
            ax=ax_map
        )

        # Plot scatter
        self.plot_avg_cases_vs_distance_scatter(
            cases_by_date=cases_by_date,
            distance_dict=distance_dict,
            linkage_matrix=linkage_matrix,
            counties=counties,
            county_number_map=county_number_map,
            cluster_threshold=cluster_threshold,
            selected_snapshots=selected_snapshots,
            ax=ax_scatter
        )

        # Keep y-axis (left) spine for dendrogram
        for spine in ["top", "right", "bottom"]:
            ax_dendro.spines[spine].set_visible(False)

        # Remove all spines for map (geographical)
        for spine in ax_map.spines.values():
            spine.set_visible(False)

        # Abbreviation legend
        ax_abbrev.axis("off")
        sorted_counties = sorted(county_number_map.items(), key=lambda x: x[1])
        n_cols = 6
        x_spacing = 0.18
        y_spacing = 0.16
        font_size = 20

        for idx, (county, num) in enumerate(sorted_counties):
            row = idx // n_cols
            col = idx % n_cols
            x = 0.02 + col * x_spacing
            y = 1.0 - row * y_spacing
            label = f"{num}: {county.title()}"
            color = self.dendo_colors.get(str(num), "black")

            ax_abbrev.text(x, y, label,
                           transform=ax_abbrev.transAxes,
                           fontsize=font_size, color=color,
                           ha="left", va="top", fontweight="medium")

        # Adjust layout to allow legend to fit
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, facecolor='white', transparent=False)
        plt.close()
        print(f"figure saved to: {save_path}")

    def plot_multiple_choropleths(self, shapefile_gdf, variable_dicts,
                                  titles, output_folder="output/maps"):
        def normalize_name(name):
            return name.lower().replace("-", " ").replace("_", " ").strip().title()

        # Prepare shapefile
        shapefile_gdf = shapefile_gdf.rename(
            columns={"NAME": "County"} if "NAME" in shapefile_gdf.columns else {})
        shapefile_gdf["County"] = shapefile_gdf["County"].apply(normalize_name)
        shapefile_gdf["County"] = shapefile_gdf["County"].replace(self.correction_map)

        os.makedirs(output_folder, exist_ok=True)

        for variable_name, var_dict in variable_dicts.items():
            # Prepare variable DataFrame
            df = pd.DataFrame.from_dict(var_dict, orient='index', columns=["Value"])
            df.index.name = "County"
            df.reset_index(inplace=True)
            df["County"] = df["County"].apply(normalize_name)

            merged = shapefile_gdf.merge(df, on="County", how="left")

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 16))
            cmap = plt.cm.coolwarm
            vmin = merged["Value"].min()
            vmax = merged["Value"].max()
            norm = Normalize(vmin=vmin, vmax=vmax)

            merged.plot(column="Value", cmap=cmap, norm=norm,
                        edgecolor="black", linewidth=0.4, ax=ax)

            # Annotate
            for _, row in merged.iterrows():
                if row.geometry and pd.notna(row["Value"]):
                    centroid = row.geometry.centroid
                    txt = ax.text(centroid.x, centroid.y, row["County"],
                                  fontsize=9.5, ha="center", va="center", color="black")
                    txt.set_path_effects([
                        patheffects.Stroke(linewidth=1.5, foreground='white'),
                        patheffects.Normal()
                    ])

            ax.axis("off")
            ax.set_xlim(shapefile_gdf.total_bounds[[0, 2]])
            ax.set_ylim(shapefile_gdf.total_bounds[[1, 3]])

            # Add horizontal colorbar
            cax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=10, length=0)
            cbar.set_label(titles[variable_name], fontsize=13, weight="bold", labelpad=8)

            # Save
            fname = os.path.join(output_folder,
                                 f"{variable_name.replace(' ', '_').lower()}_map.pdf")
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plt.savefig(fname, dpi=500)
            plt.close()
            print(f"Saved: {fname}")

    def plot_stacked_percent_bars_by_county(self, variable_dicts, output_dir,
                                            filename="county_stacked_percent.pdf"):
        county_region_map = {
            county: region for region, counties in self.region_order.items()
            for county in counties
        }

        # --- Normalize GDP and Working Population to Percentages ---
        gdp_dict = variable_dicts["GDP"]
        working_dict = variable_dicts["Working Population"]

        gdp_total = sum(gdp_dict.values())
        working_total = sum(working_dict.values())

        gdp_percent = {k: (v / gdp_total) * 100 for k, v in gdp_dict.items()}
        working_percent = {k: (v / working_total) * 100 for k, v in working_dict.items()}

        variable_dicts["GDP"] = gdp_percent
        variable_dicts["Working Population"] = working_percent

        # --- Prepare Data ---
        data_rows = []
        for county, region in county_region_map.items():
            row = {
                "County": county,
                "Region": region,
                "GDP": variable_dicts["GDP"].get(county, 0),
                "Poverty Rate": variable_dicts["Poverty Rate"].get(county, 0),
                "Working Population": variable_dicts["Working Population"].get(county, 0),
                "TV Access": variable_dicts["TV Access"].get(county, 0)
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df["RegionOrder"] = df["Region"].map(lambda r: list(self.region_order).index(r))
        df = df.sort_values(by=["RegionOrder", "Region", "County"]).reset_index(drop=True)

        mpl.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "font.size": 15
        })

        categories = ["GDP", "Poverty Rate", "Working Population", "TV Access"]
        colors = ["red", "cyan", "purple", "orange"]

        intra_spacing = 0.001
        inter_region_spacing = 0.002
        bar_width = intra_spacing * 0.7

        x_positions = []
        county_labels = []
        region_centers = []
        county_x_map = {}
        x = 0

        for region, counties in self.region_order.items():
            if not counties:
                continue
            start_x = x
            for county in counties:
                county_x_map[county] = x
                x_positions.append(x)
                county_labels.append(county)
                x += intra_spacing
            end_x = x - intra_spacing
            center = (start_x + end_x) / 2
            region_centers.append((center, region))
            x += inter_region_spacing

        df["x_plot"] = df["County"].map(county_x_map)
        df = df.sort_values(by="x_plot").reset_index(drop=True)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 4))  # Smaller, horizontal focus
        bottoms = np.zeros(len(df))

        for i, cat in enumerate(categories):
            ax.bar(df["x_plot"], df[cat], bottom=bottoms,
                   color=colors[i],
                   width=bar_width,
                   edgecolor='white',
                   linewidth=0.4,
                   label=cat)
            bottoms += df[cat].values

        ax.set_xticks(df["x_plot"])
        ax.set_xticklabels(df["County"], rotation=90, fontsize=6)
        ax.set_ylabel("Percentage (%)", fontsize=15, fontweight='bold')
        ax.set_ylim(0, bottoms.max() * 1.12)

        ax.tick_params(axis='y', which='both', length=3, width=0.8, direction='out')
        ax.yaxis.set_tick_params(labelsize=8)

        # --- Region Labels (Bottom-Centered & Rotated) ---
        for center, region in region_centers:
            ax.text(center, -bottoms.max() * 0.38, region,
                    ha='center', va='top', rotation=20,
                    fontsize=8, fontweight='bold', clip_on=False)

        handles, labels = ax.get_legend_handles_labels()

        plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.38)

        fig.legend(
            handles, labels,
            fontsize=10,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
            frameon=False
        )
        plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"plot saved to: {filepath}")
