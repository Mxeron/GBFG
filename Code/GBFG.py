import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist


def calculate_center_and_radius(gb):    
    data_no_label = gb[:,:]
    center = data_no_label.mean(axis=0)
    radius = np.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius


def splits(gb_list, num, k=2):
    gb_list_new = []
    for gb in gb_list:
        p = gb.shape[0]
        if p < num:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb, k))
    return gb_list_new


def splits_ball(gb, k):
    ball_list = []
    len_no_label = np.unique(gb, axis=0)
    if len_no_label.shape[0] < k:
        k = len_no_label.shape[0]
    label = k_means(X=gb, n_clusters=k, n_init=1, random_state=8)[1]
    for single_label in range(0, k):
        ball_list.append(gb[label == single_label, :])
    return ball_list


def assign_points_to_closest_gb(data, gb_centers):
    assigned_gb_indices = np.zeros(data.shape[0])
    for idx, sample in enumerate(data):
        t_idx = np.argmin(np.sqrt(np.sum((sample - gb_centers) ** 2, axis=1)))
        assigned_gb_indices[idx] = t_idx
    return assigned_gb_indices.astype('int')


def fuzzy_similarity(t_data, sigma=1.0, k=2):
    t_n, t_m = t_data.shape
    gb_list = [t_data]
    num = np.ceil(t_n ** 0.5)
    while True:
        ball_number_1 = len(gb_list)
        gb_list = splits(gb_list, num=num, k=k)
        ball_number_2 = len(gb_list)
        if ball_number_1 == ball_number_2:
            break
    gb_center = np.zeros((len(gb_list), t_m))
    for idx, gb in enumerate(gb_list): 
        gb_center[idx], _ = calculate_center_and_radius(gb)
    point_to_gb = assign_points_to_closest_gb(t_data, gb_center)

    tp = 1 / (1 + cdist(gb_center, gb_center) / t_m)
    tp[tp < sigma] = 0 
    return tp, point_to_gb


def GBFG(data, sigma=1):
    n, m = data.shape
    LA = np.arange(m)
    OD = np.zeros((n, m), dtype=np.float32)
    point_fs = np.zeros((n, n))
    for idx1, l1 in enumerate(LA):
        gb_fs, point_to_gb = fuzzy_similarity(data[:,[l1]], sigma, k=2)
        for s in range(n):
            for t in range(s + 1):
                point_fs[s, t] = gb_fs[point_to_gb[s], point_to_gb[t]]
                point_fs[t, s] = point_fs[s, t]
        rel_mat_k_l, ic = np.unique(point_fs, axis=0, return_inverse=True)
        n_items = rel_mat_k_l.shape[0]
        for i in range(n_items):
            i_tem = np.where(ic == i)[0]
            OD[i_tem, idx1] = rel_mat_k_l[i].mean()
    OD = 1 - np.cbrt(OD)
    OF = np.mean(OD, axis=1)
    return OF