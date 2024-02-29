from scipy import spatial
from scipy.spatial.transform import Rotation as R
import numpy as np

T_lid2cam = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[(1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 0, 1920 / 2], [0, (1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 1080 / 2], [0, 0, 1]])

def estimate_kernel_correlation(kdtree: spatial.KDTree, pcl, params, sigma, k):
    # apply the given rotational parameters
    T_pcl = R.from_rotvec(params[:3], degrees=False).as_matrix() @ pcl[:, :3].T
    # project points
    T_pcl = T_lid2cam[:3, :3] @ T_pcl + T_lid2cam[:3, 3:]
    proj = K @ T_pcl
    proj = proj[:2, proj[2] > 0] / proj[2, proj[2] > 0] 
    # clip points outside of the image plane
    proj = proj[:,np.logical_and(np.logical_and(proj[1] >=0, proj[1] < 1080), np.logical_and(proj[0] >=0, proj[0] < 1920))]
    # estimate kernel correlation
    dist, _ = kdtree.query(proj.T, k=k)
    return -np.sum(np.exp((- (dist ** 2) / (2 * sigma ** 2)).flatten()))