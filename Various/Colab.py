from scipy.sparse import dia
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons, make_gaussian_quantiles,make_s_curve, make_regression, make_swiss_roll
from scipy.stats import kde
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from numpy import quantile, where, random, percentile, logical_or
from collections import Counter
from sklearn.metrics import silhouette_score, f1_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import random
from scipy import ndimage
from scipy.ndimage import maximum_filter
from skimage.filters import threshold_otsu
import cv2
from skimage import measure
from matplotlib.path import Path
from matplotlib.patches import Polygon
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pandas as pd
from sklearn.preprocessing import StandardScaler




def determine_epsilon(x, min_pts):
    # Distanzen zu nächsten k Nachbarn (mit k = min_pts) werden berechnet
    neigh = NearestNeighbors(n_neighbors=min_pts)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, -1]
    # maximal 7% Rauschpunkte
    epsilon = quantile(distances, .93)
    return epsilon

def noise_detection(x, n_neighbors, contamination):
    # Apply LOF
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_scores = -lof_model.fit_predict(x)
    lof_thresh = quantile(lof_scores, .999)
    lof_index = where(lof_scores >= lof_thresh)
    lof_values = x[lof_index]

    # IQR - Normalverteilung
    # Perzentile berechnen
    Q1 = percentile(x, 25, axis=0)
    Q3 = percentile(x, 75, axis=0)
    IQR = Q3 - Q1

    # welche Werte sind Ausreißer?
    iqr_mask = logical_or((x < Q1 - 1.5 * IQR), (x > Q3 + 1.5 * IQR))
    # Ausreißer werden aus Datenextrahiert
    iqr_indices = where(iqr_mask)
    iqr_values = x[iqr_indices[0]]

    min_samples_range = range(2 * x.shape[1], 3 * x.shape[1])

    # Find best parameters for DBSCAN
    best_sil_score = -1
    best_min_samples = None
    best_epsilon = None
    for min_samples in min_samples_range:
        epsilon = determine_epsilon(x, min_samples)
        model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(x)
        labels = model.labels_

        if len(set(labels)) > 2:
            silhouette_avg = silhouette_score(x[labels != -1], labels[labels != -1])

            if silhouette_avg > best_sil_score:
                best_sil_score = silhouette_avg
                best_min_samples = min_samples
                best_epsilon = epsilon
        else:
            best_epsilon = epsilon
            best_min_samples = min_samples

    model = DBSCAN(eps=best_epsilon, min_samples=best_min_samples).fit(x)
    labels = model.labels_
    index = where(labels == -1)

    # alles kombinieren
    all_outliers = np.unique(np.concatenate([index[0], lof_index[0], iqr_indices[0]]))
    all_outlier_values = x[all_outliers]
    # plot_data(x, all_outlier_values, "Ausreißererkennung")
    return best_min_samples, best_epsilon, all_outliers

# beste Parameter für LOF, probiert mögliche Parameter für LOF aus und bewertet Ergebnisse mittels F1-Score
def grid_search_lof(x, y):
    best_score = -1
    best_params = {'n_neighbors': None, 'contamination': None}

    for n_neighbors in range(10, 25):
        for contamination in [0.01, 0.02, 0.03]:
            lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            y_pred = lof_model.fit_predict(x)
            y_pred = [1 if x == -1 else 0 for x in y_pred]
            score = f1_score(y, y_pred, average='macro')

            if score > best_score:
                best_score = score
                best_params['n_neighbors'] = n_neighbors
                best_params['contamination'] = contamination
    return best_params, best_score

def outlier_removal(x, y):
    best_params, best_score = grid_search_lof(x, y)
    best_min_samples, best_epsilon, all_outliers = noise_detection(x, best_params['n_neighbors'], best_params['contamination'])
    outlier_mask = np.ones(x.shape[0], dtype=bool)
    outlier_mask[all_outliers] = False
    x_without_outliers = x[outlier_mask]
    y_without_outliers = y[outlier_mask]
    return x_without_outliers, y_without_outliers

# Maximas finden

def plot_3d_and_2d(data):
    density = kde.gaussian_kde(data.T)
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(density(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
    Z_smooth = gaussian_filter(Z, sigma=1)


    maxima_indices = []

    maxima_indices = peak_local_max(Z_smooth, min_distance=1, threshold_rel=0.7)

    if len(maxima_indices) == 0:
      filter_size = 80
      max_filtered = maximum_filter(Z_smooth, size=filter_size)
      maxima = (Z_smooth == max_filtered)
      maxima_indices = np.argwhere(maxima)

    if len(maxima_indices) == 0:
        mass_center = ndimage.measurements.center_of_mass(Z_smooth)
        maxima_indices = [np.round(mass_center).astype(int)]


    '''
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)
    average_density = np.mean(Z_smooth)
    ax.plot_surface(X, Y, np.full_like(Z_smooth, average_density), alpha=0.3, color="gray", label="Average Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    '''

    all_maximas = []
    for maxima_coords in maxima_indices:
        i, j = maxima_coords
        x_coord = X[i, j]
        y_coord = Y[i, j]
        z_coord = Z_smooth[i, j]
        all_maximas.append([x_coord, y_coord, z_coord])
        # ax.scatter(x_coord, y_coord, z_coord, color="r", s=50)
    # plt.show()

    '''
    plt.scatter(data[:, 0], data[:, 1], c="black", s=10)
    for maxima_coords in maxima_indices:
        i, j = maxima_coords
        plt.scatter(X[i, j], Y[i, j], color="r", s=50)
    plt.title("2D Plot mit Ausgangspunkten und Maxima")
    plt.show()
    '''

    return maxima_indices, X, Y, Z, Z_smooth, all_maximas



def get_contours_with_factor(factor, X, Y, Z):
    mean_z = np.mean(Z.ravel())
    contours = plt.contour(X, Y, Z, [factor * mean_z])
    contour_paths = contours.collections[0].get_paths()
    all_contour_points = []
    for path in contour_paths:
        vertices = path.vertices
        contour_points = vertices.tolist()
        all_contour_points.append(contour_points)

    return all_contour_points

def get_silhouette_score_for_contours(contour_points, data_points, labels):
    simplified_polygons = [Polygon(contour).simplify(0.1, preserve_topology=False) for contour in contour_points]
    cluster_labels = []
    for point in data_points:
        for i, polygon in enumerate(simplified_polygons):
            if polygon.contains(Point(point)):
                cluster_labels.append(i)
                break
        else:
            cluster_labels.append(-1)

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        return -1

    return silhouette_score(data_points, cluster_labels)

def find_best_factor(factors, X, Y, Z, data_points, labels):
    best_factor = None
    best_silhouette_score = -1
    best_contour_points = None

    for factor in factors:
        contour_points = get_contours_with_factor(factor, X, Y, Z)
        s = get_silhouette_score_for_contours(contour_points, data_points, labels)
        if s >= best_silhouette_score:
            best_silhouette_score = s
            best_factor = factor
            best_contour_points = contour_points

    return best_contour_points, best_factor

padding_percentage = 0.2

def prepare_data_and_create_histogram(data, padding_percentage, bins=[64, 64]):
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * padding_percentage
    x_max += x_range * padding_percentage
    y_min -= y_range * padding_percentage
    y_max += y_range * padding_percentage

    # Erstelle das 2D Histogramm
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=[[x_min, x_max], [y_min, y_max]])

    return hist, x_min, x_max, y_min, y_max

def process_histogram(hist, sigma):
    dx = ndimage.sobel(hist, 0)
    dy = ndimage.sobel(hist, 1)
    mag = np.hypot(dx, dy)

    smoothed_mag = gaussian_filter(mag, sigma=sigma)


    '''
    plt.figure()
    plt.imshow(smoothed_mag, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.title('2D Histogram after Sobel Operator')
    plt.show()
    '''

    binary_image = smoothed_mag > 1
    contours = measure.find_contours(binary_image, 0.8)

    return smoothed_mag, contours

def scale_and_plot_contours(contours, x_min, x_max, y_min, y_max, bins=[64, 64]):
    scaled_contours = []
    for contour in contours:
        scaled_contour = np.empty_like(contour)
        scaled_contour[:, 0] = x_min + contour[:, 0] * (x_max - x_min) / bins[0]
        scaled_contour[:, 1] = y_min + contour[:, 1] * (y_max - y_min) / bins[1]

        scaled_contours.append(scaled_contour)

    contour_points = [arr.tolist() for arr in scaled_contours]

    # plt.figure()
    for contour in contour_points:
        x = [point[0] for point in contour]
        y = [point[1] for point in contour]
        # plt.plot(x, y)
    # plt.gca().invert_yaxis()
    # plt.show()

    return contour_points



def entscheide_ueber_beste_methode(contour_points_method1, contour_points_method2, data_points, labels):
    contour_points_method2 = [[[p[0], p[1]] for p in contour] for contour in contour_points_method2]

    f1_scores = []
    cluster_labels_list = []
    for cluster_points_list in [contour_points_method1, contour_points_method2]:
        cluster_labels = []
        # Polygone vereinfachen, damit geringere Zeitkomplexität
        simplified_polygons = [Polygon(contour).simplify(0.1, preserve_topology=False) for contour in cluster_points_list]
        for point in data_points:
            for i, polygon in enumerate(simplified_polygons):
                if polygon.contains(Point(point)):
                    cluster_labels.append(i)
                    break
            else:
                cluster_labels.append(-1)

        f1 = f1_score(labels, cluster_labels, average='macro')
        f1_scores.append(f1)
        cluster_labels_list.append(cluster_labels)

    if f1_scores[0] > f1_scores[1]:
        return contour_points_method1, "method1", cluster_labels_list[0]
    else:
        return contour_points_method2, "method2", cluster_labels_list[1]
    
sigma_values = [0.7, 1, 1.5, 3]

def find_best_sigma(sigmas, contour_points_method1, data_points, labels):
    hist, x_min, x_max, y_min, y_max = prepare_data_and_create_histogram(data_points, padding_percentage)
    best_sigma = None
    best_f1_score = 0
    best_contour_points = None

    for sigma in sigmas:
        _, contours = process_histogram(hist, sigma)
        contour_points_method2 = scale_and_plot_contours(contours, x_min, x_max, y_min, y_max)
        contour_points, method, cluster_labels = entscheide_ueber_beste_methode(contour_points_method1, contour_points_method2, data_points, labels)

        if method == "method2":
            f1 = f1_score(labels, cluster_labels, average='macro')
            print("F1: ", f1)
            if f1 >= best_f1_score:
                best_f1_score = f1
                best_sigma = sigma
                best_contour_points = contour_points

    if best_sigma == None:
        hist, x_min, x_max, y_min, y_max = prepare_data_and_create_histogram(data_points, padding_percentage)
        _, contours = process_histogram(hist, 1.5)
        contour_points_method2 = scale_and_plot_contours(contours, x_min, x_max, y_min, y_max)
        best_contour_points = contour_points_method2

    print("Best sigma:", best_sigma)

    return best_contour_points, best_sigma

def euklidische_distanz(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def berechne_konturpunkte_fuer_maxima(all_maximas, contour_points_list):
    konturpunkte_fuer_maxima = [[] for _ in range(len(all_maximas))]
    zugewiesene_konturen = set()

    for i, maxima in enumerate(all_maximas):
        x_max, y_max, _ = maxima
        maxima_punkt = np.array([[x_max, y_max]])

        for j, contour_points in enumerate(contour_points_list):
            if j in zugewiesene_konturen:
                continue

            path = Path(contour_points)
            if path.contains_points(maxima_punkt):
                konturpunkte_fuer_maxima[i] = contour_points
                zugewiesene_konturen.add(j)
                break

    return konturpunkte_fuer_maxima


def berechne_distanzen(all_maximas, konturpunkte_fuer_maxima):
    distanzen = []
    for i, maxima in enumerate(all_maximas):
        x_max, y_max, _ = maxima
        konturpunkte = konturpunkte_fuer_maxima[i]
        distanzen_maximum = []

        for contour_point in konturpunkte:
            x_contour, y_contour = contour_point
            distanz = euklidische_distanz([x_max, y_max], [x_contour, y_contour])
            distanzen_maximum.append(distanz)

        distanzen.append(distanzen_maximum)

    distanzen = [d for d in distanzen if d]
    return distanzen

def berechne_mittelpunkt(contour_points):
    # Entferne die Verschachtelung der Liste
    flat_list = [item for sublist in contour_points for item in sublist]

    sum_x = sum(p[0] for p in flat_list)
    sum_y = sum(p[1] for p in flat_list)
    n = len(flat_list)

    return (sum_x / n, sum_y / n)


def berechne_distanzen_vom_mittelpunkt(mittelpunkt, contour_points):
    distanzen = [[euklidische_distanz(mittelpunkt, punkt) for punkt in contour] for contour in contour_points]
    return distanzen


def berechne_exzentrizitaet(contour_points):
    exzentrizitaeten = []

    for contour in contour_points:
        contour_array = np.array(contour, dtype=np.float32)

        moments = cv2.moments(contour_array)

        if len(contour) >= 5:
            (x,y),(MA,ma),angle = cv2.fitEllipse(contour_array)

            exzentrizitaet = np.sqrt(1 - (MA/ma)**2)
            exzentrizitaeten.append(exzentrizitaet)

    return np.mean(exzentrizitaeten)

def berechne_dichte_in_kontur(contour_points_list, data_points):
    gesamt_punkte = len(data_points)
    punkte_in_kontur = 0
    gewichtete_dichte = 0

    for contour in contour_points_list:
        polygon = Polygon(contour)
        punkte_in_polygon = [point for point in data_points if polygon.contains(Point(point[0], point[1]))]
        punkte_in_kontur += len(punkte_in_polygon)
        if polygon.area > 0:
            dichte_in_polygon = len(punkte_in_polygon) / polygon.area
            gewichtete_dichte += dichte_in_polygon

    if punkte_in_kontur > 0:
        gewichtete_dichte *= (punkte_in_kontur / gesamt_punkte)

    return gewichtete_dichte

factors = [1.0, 1.5, 3.0]

def calc_unscaled_features(X, y):
    data = X
    x_without_outliers, y_without_outliers = outlier_removal(X, y)
    maxima_indices, X_, Y, Z, Z_smooth, all_maximas = plot_3d_and_2d(x_without_outliers)

    contour_points_method1, _ = find_best_factor(factors, X_, Y, Z, x_without_outliers, y_without_outliers)

    # contour_points_method1 = plot_contour(maxima_indices, X_, Y, Z, Z_smooth)
    hist, x_min, x_max, y_min, y_max = prepare_data_and_create_histogram(x_without_outliers, padding_percentage)

    sigma_values = [0.7, 1, 1.5, 3]
    contour_points_method2, best_sigma = find_best_sigma(sigma_values, contour_points_method1, x_without_outliers, y_without_outliers)

    contour_points, method, _ = entscheide_ueber_beste_methode(contour_points_method1, contour_points_method2, x_without_outliers, y_without_outliers)

    konturpunkte_fuer_maxima = berechne_konturpunkte_fuer_maxima(all_maximas, contour_points)
    distanzen = berechne_distanzen(all_maximas, konturpunkte_fuer_maxima)

    mittelpunkt_kontur = berechne_mittelpunkt(contour_points)
    distanzen_vom_mittelpunkt = berechne_distanzen_vom_mittelpunkt(mittelpunkt_kontur, contour_points)

    average_density_in_contour = berechne_dichte_in_kontur(contour_points, x_without_outliers)

    def find_shape_ratio(distanzen):
        return min(distanzen) / max(distanzen)

    def find_standard_deviation(distanzen):
        return np.std(distanzen)

    def calc_total_points(data):
        return len(data)

    def calc_mean_distance_to_centroid(data, maxima):
        distances = []
        for point in data:
            min_distance = min(euklidische_distanz(point, m[:2]) for m in maxima)
            distances.append(min_distance)
        return np.mean(distances), np.std(distances)

    def calc_mean_k_nearest(data, k=5):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(data)
        distances, _ = neigh.kneighbors(data)
        mean_distances = np.mean(distances, axis=1)
        return np.mean(mean_distances), np.std(mean_distances)

    def calc_kurtosis_and_skewness(data):
        kurtosis_vals = stats.kurtosis(data)
        skewness_vals = stats.skew(data)
        return np.mean(kurtosis_vals), np.std(kurtosis_vals), np.mean(skewness_vals), np.std(skewness_vals)

    shape_ratios = []
    std_devs = []
    for distanzen_ in distanzen:
        shape_ratio = find_shape_ratio(distanzen_)
        std_dev = find_standard_deviation(distanzen_)
        shape_ratios.append(shape_ratio)
        std_devs.append(std_dev)

    shape_ratios_middle = []
    std_devs_middle = []
    for distanzen_ in distanzen_vom_mittelpunkt:
        shape_ratio_middle = find_shape_ratio(distanzen_)
        std_dev_middle = find_standard_deviation(distanzen_)
        shape_ratios_middle.append(shape_ratio_middle)
        std_devs_middle.append(std_dev_middle)

    # Radien von Maximum zu Kontur
    mean_shape_ratio = np.mean(shape_ratios)
    std_shape_ratio = np.std(shape_ratios)
    mean_std_dev = np.mean(std_devs)
    std_std_dev = np.std(std_devs)

    # Radien von Mittelpunkt der Kontur zu Kontur
    mean_shape_ratio_middle = np.mean(shape_ratios_middle)
    std_shape_ratio_middle = np.std(shape_ratios_middle)
    mean_std_dev_middle = np.mean(std_devs_middle)
    std_std_dev_middle = np.std(std_devs_middle)

    total_points = calc_total_points(data)
    mean_distance_to_centroid, std_dev_distance_to_centroid = calc_mean_distance_to_centroid(data, all_maximas)
    mean_k_nearest, std_k_nearest = calc_mean_k_nearest(data)
    mean_kurtosis, std_kurtosis, mean_skewness, std_skewness = calc_kurtosis_and_skewness(data)

    # mean_shape_ratio: Durchschnitt des Verhältnisses der Form der Datenpunkte, vom Maximum aus betrachtet
    # std_shape_ratio: Standardabweichung des Verhältnisses der Form der Datenpunkte, vom Maximum aus
    # mean_std_dev: Durchschnittliche Standardabweichung der Datenpunkte
    # std_std_dev: Standardabweichung der Standardabweichung der Datenpunkte
    # mean_shape_ratio_middle: Durchschnitt des Verhältnisses der Form von Formmittelpunkt aus betrachtet
    # std_shape_ratio_middle: Standardabweichung des Verhältnisses der Form von Formmittelpunkt aus betrachtet
    # mean_std_dev_middle: Durchschnittliche Standardabweichung der Distanzen vom  Maximum
    # std_std_dev_middle: Standardabweichung der Standardabweichung der Distanzen vom  Maximum

    # total_points: Gesamtzahl der Datenpunkte
    # mean_distance_to_centroid: Durchschnittlicher Abstand der Datenpunkte zum Maximum
    # std_dev_distance_to_centroid: Standardabweichung des Abstands der Datenpunkte zum Maximum

    # mean_k_nearest: Durchschnittlicher Abstand zu den k nächsten Nachbarn (k=5)
    # std_k_nearest: Standardabweichung des Abstands zu den k nächsten Nachbarn

    # mean_kurtosis: Durchschnittliche Kurtosis (Spitzheit) der Datenpunkte
    # std_kurtosis: Standardabweichung der Kurtosis der Datenpunkte
    # mean_skewness: Durchschnittliche Schiefe der Datenpunkte
    # std_skewness: Standardabweichung der Schiefe der Datenpunkte

    # average_density_in_contour: Durchschnittliche Dichte der Datenpunkte innerhalb der Kontur


    unscaled_features = np.array([
        mean_shape_ratio, std_shape_ratio, mean_std_dev, std_std_dev, mean_shape_ratio_middle, std_shape_ratio_middle,
        mean_std_dev_middle, std_std_dev_middle, total_points, mean_distance_to_centroid, std_dev_distance_to_centroid,
        mean_k_nearest, std_k_nearest, mean_kurtosis, std_kurtosis, mean_skewness,
        std_skewness, average_density_in_contour])


    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(unscaled_features.reshape(-1, 1))

    return unscaled_features, scaled_features, scaler

class DataAugmentor:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.n_samples, self.n_features = data.shape

    def remove_points(self, n: int):
        indices = np.random.choice(self.n_samples, size=n, replace=False)
        return np.delete(self.data, indices, axis=0)

    def add_points(self, n: int):
        indices = np.random.choice(self.n_samples, size=n)
        additional_points = self.data[indices] + 0.05 * np.random.randn(
            n, self.n_features
        )
        return np.vstack([self.data, additional_points])

    def scale_data(self, scale_factor: float):
        return self.data * scale_factor

    def shift_points(self, shift_factor: float):
        shifts = (
            (np.random.rand(self.n_samples, self.n_features) - 0.5) * 2 * shift_factor
        )
        return self.data + shifts

    def rotate_around_centroid(self, theta: float):
        # Berechne den Schwerpunkt der Daten
        centroid = np.mean(self.data, axis=0)

        # Verschiebe die Daten, so dass der Schwerpunkt im Ursprung liegt
        shifted_data = self.data - centroid

        # Rotationsmatrix
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Drehe die Punkte
        rotated_data = np.dot(shifted_data, R.T)

        # Verschiebe die Daten zurück
        return rotated_data + centroid

    def augment(self):
        result_data = self.data.copy()

        # Entfernen von Datenpunkten
        remove_factor = np.random.randint(1, self.n_samples)
        result_data = self.remove_points(remove_factor)

        # Hinzufügen von Datenpunkten
        add_factor = np.random.randint(1, self.n_samples)
        result_data = self.add_points(add_factor)

        # Skalieren der Daten
        scale_factor = 0.5 + np.random.rand()
        result_data = self.scale_data(scale_factor)

        # Verschieben der Datenpunkte
        shift_factor = 0.05
        result_data = self.shift_points(shift_factor)

        # Rotation um einen zufälligen Winkel | funktioniert nur für 2D Daten
        theta = np.random.rand() * 2 * np.pi  # Zufälliger Winkel zwischen 0 und 2π
        result_data = self.rotate_around_centroid(theta)

        return result_data
    

# Generatoren

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_s_curve, make_gaussian_quantiles
from sklearn.preprocessing import StandardScaler


# n_samples_range = [200, 270, 350, 500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]. 17
n_samples_range = [200]
# n_clusters_range = [2, 3, 4, 5, 6]. 5
n_clusters_range = [2, 3]
# noise_range = [0.1, 0.2, 0.3]. 3
noise_range = [0.1]
# n_samples_range_moons = [200, 270, 350, 500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]
n_samples_range_moons = [200]
# noise_range_moons = [0.1, 0.2, 0.3]
noise_range_moons = [0.1]

n_augment = 1

# n_iter_per_combination = 5
n_iter_per_combination = 1

datasets = []

best_algorithms_blobs = ["KMeans"] * n_iter_per_combination
best_algorithms_circles = ["DBSCAN"] * n_iter_per_combination
best_algorithms_moons = ["DBSCAN"] * n_iter_per_combination
best_algorithms_gaussian_quantiles = ["GMM"] * n_iter_per_combination

scaler = StandardScaler()


for n_samples in n_samples_range:
    for n_clusters in n_clusters_range:
        for noise in noise_range:
            for i in range(n_iter_per_combination):
                random_state = np.random.randint(10000)

                X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=noise, random_state=random_state)
                X_scaled = scaler.fit_transform(X)
                data_augmentor = DataAugmentor(X_scaled)
                for j in range(n_augment):
                    augmentedData = data_augmentor.augment()
                    datasets.append((augmentedData, y, best_algorithms_blobs[i]))

                datasets.append((X_scaled, y, best_algorithms_blobs[i]))

                X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
                X_scaled = scaler.fit_transform(X)
                data_augmentor = DataAugmentor(X_scaled)
                for j in range(n_augment):
                    augmentedData = data_augmentor.augment()
                    datasets.append((augmentedData, y, best_algorithms_blobs[i]))
                datasets.append((X_scaled, y, best_algorithms_circles[i]))


for n_samples in n_samples_range_moons:
    for noise in noise_range_moons:
        for i in range(n_iter_per_combination):
            random_state = np.random.randint(10000)

            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
            X_scaled = scaler.fit_transform(X)
            data_augmentor = DataAugmentor(X_scaled)
            for j in range(n_augment):
                augmentedData = data_augmentor.augment()
                datasets.append((augmentedData, y, best_algorithms_blobs[i]))
            datasets.append((X_scaled, y, best_algorithms_moons[i]))

            X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=2, random_state=random_state)
            X_scaled = scaler.fit_transform(X)
            data_augmentor = DataAugmentor(X_scaled)
            for j in range(n_augment):
                augmentedData = data_augmentor.augment()
                datasets.append((augmentedData, y, best_algorithms_blobs[i]))
            datasets.append((X_scaled, y, best_algorithms_gaussian_quantiles[i]))


feature_list = []
label_list = []

counter = 0
for x, y, best_algorithm in datasets:
    x = np.array(x)
    y = np.array(y)
    unscaled_features, scaled_features, _ = calc_unscaled_features(x, y)
    print(unscaled_features)
    counter += 1
    print("Durchlauf: ", counter)
    feature_list.append(scaled_features)
    label_list.append(best_algorithm)

features_array = np.array(feature_list)
labels_array = np.array(label_list)

num_datasets_to_plot = 100

for i, (X, y, algorithm) in enumerate(datasets[:num_datasets_to_plot]):
    plt.figure(i)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(f'Dataset {i} - Best Algorithm: {algorithm}')

print(features_array)

features_array = features_array.squeeze()

df = pd.DataFrame(features_array, columns=[f"Feature {i+1}" for i in range(features_array.shape[1])])

df['Best Algorithm'] = labels_array

cols = ['Best Algorithm'] + [col for col in df if col != 'Best Algorithm']
df = df[cols]

df.to_excel('data.xlsx', index=False, engine='openpyxl')