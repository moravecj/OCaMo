import numpy as np
import matplotlib.pyplot as plt

def non_maxima_suppression(arr: np.ndarray, w, thr):
    sup = np.zeros_like(arr, dtype=bool)
    sup[w:-w] = arr[w:-w] > thr
    for i in range(-w, w + 1, 1):
        sup[w:-w] = np.logical_and(sup[w:-w], (arr[w:-w] >= arr[w+i:sup.shape[0]-w+i]))
    return sup

def get_corners(pcl):
    der_gaussian = lambda x: -x * np.exp(-(x*x) / 2)
    mask = der_gaussian(np.arange(-5, 6, 1))
    rd = np.sqrt(pcl[:, 0] ** 2 + pcl[:, 1] ** 2 + pcl[:, 2] ** 2)
    # Choose closer corner
    r_right = np.hstack([np.arange(1, rd.shape[0]), rd.shape[0]-1]) 
    r_left = np.hstack([0, np.arange(0, rd.shape[0] - 1)])
    r_right_idx = rd[r_right] < rd - 0.5
    r_left_idx = rd[r_left] < rd - 0.5
    r_closest = np.arange(0, rd.shape[0])
    r_closest[r_right_idx] = r_right[r_right_idx]
    r_closest[r_left_idx] = r_left[r_left_idx]

    # jump in radial distance
    rd_mask = np.abs(np.convolve(rd, mask, mode='same') / np.sqrt(np.convolve(np.square(rd), np.ones(11,), mode='same')))
    rd_idx = np.zeros((pcl.shape[0],))
    rd_idx = rd_mask
    # non-maxima suppression, thresholding
    rd_idx = non_maxima_suppression(rd_idx, 2, 0.01)
    # jump in azimuth
    az = np.arctan2(pcl[:, 1], pcl[:, 0])
    az_diff = np.maximum(np.abs(np.diff(az, append=az[-1])), np.abs(np.diff(az, prepend=az[0])))

    idx = np.zeros_like(rd, dtype=bool)
    idx[az_diff > 0.1] = 1
    idx[r_closest[rd_idx]] = 1

    return pcl[idx, :]