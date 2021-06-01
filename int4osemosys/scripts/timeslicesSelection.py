import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from itertools import combinations


def time_slices_selection(tssname, timeseries_dir, ts_names, n_clusters=12, n_far_points=1, n_global_far_points=1):
    """Select time slices based on k-means and prepare time configuration file.

    Apply k-means algorithm for clustering the 8760 hours of a year into **n_cluesters** different groups.
    Each hour is distinguished the vales of the different timeseries (in **ts_names**) files which can be found at the **timeseries_dir**.
    To the cluster centers selection is added:

    (1) **n_far_points** with the highest overall distance to their own center are added to the time slice selection.

    (2) **n_global_far_points** with the highest overall  all centers are also selected as a time slices.

    Making a total of **n_clusters + n_clusters*n_far_points + n_global_far_points** time slices selected.

    The time slice weight parameter is set according to the number hours represented by the time steps
    (i.e. 1 for the "far points" and the number of elements in the cluster for the centers).
    The obtained time slices and time distributions are wirtten to an **tssname**.csv file at the **timeseries_dir**.

    Example:

        >>> from scripts import timeslicesSelection as tss
        >>> import os # relative path

        Set path and give timeseries names to be used

        >>> # Set use the timeseries dir inside the data dir
        >>> timeseries_dir = os.sep.join(["..", "data", "timeseries"])
        >>> # Select timeseries to be considered for clustering (must be on the aforementioned folder)
        >>> ts_names = ["corrected_eletricity_demand_2016.txt",
        >>>             "heat_household_demand_2016.txt",
        >>>             "Solar_availability_1004FLH.txt",
        >>>             "Solar_availability_2016.txt",
        >>>             "WindOffshore_availability_2016.txt",
        >>>             "WindOffshore_availability_4300FLH.txt",
        >>>             "WindOnshore_availability_2016.txt",
        >>>             "WindOnshore_availability_2700FLH.txt",
        >>>             "Uniform.txt"]

        Run k-means

        >>> tss.time_slices_selection("tssname", timeseries_dir, ts_names)

    Args:
        tssname (str): Name for the timeslice selection
        timeseries_dir(str): Directory where the time series files are.
                             Files must be ".txt" with 8760 entries separated by space. See examples in dir.
        ts_names(list[str]): List with the name of the timeseries to be considered for the k-means clustering.
        n_clusters (int) : n of clusters for the k-means algorithm.
        n_far_points (int): n furthest points from its own cluster center to be considered as individual timeslices.
        n_global_far_points (int): n furthest points from all clusters to be considered as individual timeslices.


    Raises:
        ValueError: File from timeseries name given in **ts_names** not found in **timeseries_dir**

    """

    # Set for True to render figures
    SHOW_PLOTS = False
    SHOW_F1 = False
    SHOW_BOX = False
    SHOW_PLANES = False


    # Check if all names are in the folder
    missing_files = [x for x in ts_names if x not in os.listdir(timeseries_dir)]
    if len(missing_files) != 0:
        raise ValueError("Timeseries %s not in %s" % (missing_files, timeseries_dir))

    # load timeseries. Final Shape: i_hours X j_timeseries
    ts = dict()
    ts_data = np.empty(shape=(8760, len(ts_names)))

    for fname in ts_names:
        data = np.loadtxt(os.sep.join([timeseries_dir, fname]))
        ts[fname] = data
        ts_data[:, ts_names.index(fname)] = data

    # make dataframe!
    ts = pd.DataFrame.from_dict(ts)

    # start scaler and scale data
    scaler = MinMaxScaler()
    scaler.fit(ts_data)
    ts_data_scaled = scaler.transform(ts_data)

    # Build x arrays for k-means.
    X = ts_data_scaled

    # Run k-means
    kmeans = KMeans(init="random", n_clusters=n_clusters, max_iter=300)
    kmeans.fit(X)

    # Get the indices of the points for each corresponding cluster
    cluster_dict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    cluster_dict = dict(cluster_dict)

    # Compute distance to clusters
    #  shape (n_samples, n_features) ->  shape(n_samples, n_clusters)
    distances = kmeans.transform(X)

    # find TS that is the further away from all clusters. --> Higher Norm
    # Compute norm
    norms = np.linalg.norm(distances, axis=1)

    # Find max dist to each cluster
    furthest_in_cluster = dict()
    for c in cluster_dict.keys():
        furthest_in_cluster[c] = int(np.where(distances[:, c] == np.amax(distances[cluster_dict[c], c]))[0])

    # find max in all clusters
    furthest = int(np.where(np.amax(norms) == norms)[0])


    colors = ['#2AE6CA', '#BEBD7F', '#1C542D', '#4C2F27', '#E5BE01', '#26252D', '#A98307', '#F80000', '#D36E70', '#7FB5B5',
              '#17BF46',
              '#E1CF4F', '#E6D690', '#BDEFD1', '#F5D033', '#F9F22B', '#9E9764', '#999950', '#2AE6CA', '#BEBD7F', '#1C542D',
              '#4C2F27', '#E5BE01', '#26252D', '#A98307', '#F80000', '#D36E70', '#7FB5B5', '#17BF46',
              '#E1CF4F', '#E6D690', '#BDEFD1', '#F5D033', '#F9F22B', '#9E9764', '#999950', '#2AE6CA', '#BEBD7F', '#1C542D',
              '#4C2F27', '#E5BE01', '#26252D', '#A98307', '#F80000', '#D36E70', '#7FB5B5', '#17BF46',
              '#E1CF4F', '#E6D690', '#BDEFD1', '#F5D033', '#F9F22B', '#9E9764', '#999950', '#2AE6CA', '#BEBD7F', '#1C542D',
              '#4C2F27', '#E5BE01', '#26252D', '#A98307', '#F80000', '#D36E70', '#7FB5B5', '#17BF46',
              '#E1CF4F', '#E6D690', '#BDEFD1', '#F5D033', '#F9F22B', '#9E9764', '#999950', '#2AE6CA', '#BEBD7F', '#1C542D',
              '#4C2F27', '#E5BE01', '#26252D', '#A98307', '#F80000', '#D36E70', '#7FB5B5', '#17BF46',
              '#E1CF4F', '#E6D690', '#BDEFD1', '#F5D033', '#F9F22B', '#9E9764', '#999950']

    if SHOW_PLOTS:
        # Heatmap with cluster centers
        f = go.Figure()
        values = np.array(kmeans.cluster_centers_).transpose()
        f.add_trace(go.Heatmap(z=values, colorscale="Viridis", y=ts_names))
        f.show()

    if SHOW_F1:
        # Centers and variations
        ncols = 2
        plots_each_page = 4
        all_clusters = list(cluster_dict.keys())
        figures = []
        while all_clusters:
            if len(all_clusters) > plots_each_page:
                c_list = [all_clusters.pop() for idx in range(plots_each_page)]
            else:
                c_list = all_clusters
                all_clusters = []
            f2 = make_subplots(rows=math.ceil(len(c_list)/ncols), cols=ncols)
            for ii in range(len(c_list)):
                c = c_list[ii]
                #  PLot centroid
                f2.add_trace(go.Scatter(x=kmeans.cluster_centers_[c], y=ts_names, mode="markers", legendgroup="C%d" % c,
                                        name="Cluster %d | Nel = %d" % (c, len(cluster_dict[c])),
                                        marker=dict(
                                            size=20,
                                            opacity=1,
                                            color=colors[c],
                                        )), row=ii//ncols +1 , col=(ii)%ncols+1)

                # plot elements
                for el in cluster_dict[c]:
                    if el == furthest:
                        f2.add_trace(go.Scatter(x=X[el, :], y=ts_names, mode='markers', legendgroup="C%d" % c,
                                                name="Global Furthest",
                                                showlegend=True,
                                                marker=dict(
                                                    size=20,
                                                    opacity=1,
                                                    color='black',
                                                    symbol="diamond-wide"
                                                )), row=ii//ncols +1, col=(ii)%ncols+1)

                    elif el == furthest_in_cluster[c]:
                        f2.add_trace(go.Scatter(x=X[el, :], y=ts_names, mode='markers', legendgroup="C%d" % c, name="Furthest",
                                                showlegend=True,
                                                marker=dict(
                                                    size=20,
                                                    opacity=1,
                                                    color=colors[c],
                                                    symbol="diamond"
                                                )), row=ii // ncols + 1, col=ii% ncols + 1)
                    else:
                        f2.add_trace(go.Scatter(x=X[el, :], y=ts_names, mode='markers', legendgroup="C%d" % c,
                                                showlegend=False,
                                                marker=dict(
                                                    size=5,
                                                    opacity=0.3,
                                                    color=colors[c]
                                                )), row=ii//ncols +1, col=ii % ncols + 1)
            #f2.show()
            figures.append(f2)

        for f in figures:
            f.write_image("k-means-TimesliceSelection-%dof%d.pdf" % (figures.index(f), len(figures)))

    if SHOW_BOX:
        # Box Plots
        ncols = 4
        f3 = make_subplots(rows=math.ceil(n_clusters // ncols), cols=ncols)
        for c in cluster_dict.keys():
            #  Plot each dim
            dims =X.shape[1]
            for jj in range(dims):
                f3.add_trace(go.Box(x=X[cluster_dict[c], jj],legendgroup=c,
                                    name=ts_names[jj], marker_color=colors[c], boxmean=True),
                                    row=c//ncols +1 , col=(c)%ncols+1)
        f3.show()

    if SHOW_PLANES:
        all_combinations = combinations(ts_names, 2)
        all_combinations = list(all_combinations)
        plots_each_page = 6
        while all_combinations:
            if len(all_combinations) > plots_each_page:
                com = [all_combinations.pop() for idx in range(plots_each_page)]
            else:
                com = all_combinations
                all_combinations = []
            ncols = 3
            f4 = make_subplots(rows=math.ceil(len(com)/ncols), cols=ncols)
            for ii in range(len(com)):
                x_index = ts_names.index(com[ii][0])
                y_index = ts_names.index(com[ii][1])
                for c in cluster_dict.keys():
                    f4.add_trace(go.Scatter(x=X[cluster_dict[c], x_index], y=X[cluster_dict[c], y_index],mode = 'markers', marker_color=colors[c], legendgroup=c),
                                 row=ii//ncols +1 , col=(ii)%ncols+1)
                    f4.update_xaxes(title_text=com[ii][0], row=ii//ncols +1 , col=(ii)%ncols+1)
                    f4.update_yaxes(title_text=com[ii][1], row=ii//ncols +1 , col=(ii)%ncols+1)
            f4.show()

    # Export Results to CSV - Columns: timeslice, w(eights)
    columns = ["timeslice", "w"] + ts_names
    lines = n_clusters * (1 + n_far_points) + n_global_far_points
    data = np.zeros([lines, len(columns)])

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    XX = scaler.inverse_transform(X)

    for c in cluster_dict:
        data[n_clusters+c] = [n_clusters+c, 1] + list(XX[furthest_in_cluster[c]])
        data[c, :] = [c, len(cluster_dict[c]) - n_far_points] + list(centroids[c])

    data[n_clusters * (1 + n_far_points) + n_global_far_points - 1] = [n_clusters * (1 + n_far_points) + n_global_far_points - 1, 1] + list(XX[furthest])
    data[kmeans.labels_[furthest], 1] = data[kmeans.labels_[furthest], 1] - 1

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.sep.join([timeseries_dir, tssname+".csv"]))


if __name__ == '__main__':
    pass
