from matplotlib import pyplot as plt
import cv2
from scipy.io import loadmat
import numpy as np
from scipy import spatial
from scipy.spatial.transform import Rotation as R

from include.feature_extraction import get_corners
from include.kernel_correlation import estimate_kernel_correlation

# find update to the lowest function value
def evaluate_f_index(corners, edges_kdtree, th):
    cnt = 0
    upd = np.array([])
    min_val = np.inf
    for rx in [-0.0005, 0, 0.0005]:
        for ry in [-0.0005, 0, 0.0005]:
            for rz in [-0.0005, 0, 0.0005]:
                # estimate function for this update
                param = th.copy() + np.array([rx, ry, rz])
                f_index = estimate_kernel_correlation(edges_kdtree, corners, param, 9, 10)
                # remember best update
                if f_index < min_val:
                    min_val = f_index
                    upd = np.array([rx, ry, rz])
                cnt += 1
    # return update for the parameters
    return upd

np.random.seed(42)

data_dir = '../../example/carla_example'

T_lid2cam = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[(1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 0, 1920 / 2], [0, (1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 1080 / 2], [0, 0, 1]])

number_of_frames = 1500
# simulate a cumulative decalibration of +/- 0.0005 in each rotational parameter per frame
dec = np.zeros((6,number_of_frames))
dec[:3] = (2 * (np.random.rand(3,number_of_frames) > 0.5) - 1) * 0.0005
dec_all = np.cumsum(dec, axis=1)

allTh = np.zeros((3, number_of_frames))
th = np.zeros((3,))
# iterate over frames cyclicly
for i in range(number_of_frames):
    # get image edges
    img = plt.imread('{}/{}.jpg'.format(data_dir, str(i % 200 + 1).zfill(3)))
    edges = cv2.Canny(img, 50, 100)
    edg = np.where(edges)
    edg = np.vstack((edg[1], edg[0]))
    edg = edg[:, edg[1] > 1080 / 3]
    kdtree = spatial.KDTree(edg.T)
    # get pointcloud corners
    pcl = loadmat('{}/{}.mat'.format(data_dir, str(i % 200 + 1).zfill(3)))['points']
    cor = get_corners(pcl)[:, :3]
    # syntheticaly decalibrate the corners
    T_dec = np.vstack((np.hstack((R.from_rotvec(dec_all[:3, i], degrees=False).as_matrix(), dec_all[3:6, i:i+1])), np.array([0, 0, 0, 1])))
    cor_dec = (T_dec[:3, :3] @ cor.T + T_dec[:3, 3:]).T
    # update the parameters based on the loss's minima around
    th_upd = evaluate_f_index(cor_dec, kdtree, th)
    th += th_upd
    allTh[:, i] = th.copy()

dec_all *= 180 / np.pi
allTh *= 180 / np.pi

plt.plot(dec_all[0, :].T, 'k', label='roll, decalibration')
plt.plot(dec_all[1, :].T, 'm', label='pitch, decalibration')
plt.plot(dec_all[2, :].T, 'orange', label='yaw, decalibration')
plt.plot(-allTh.T[:, 0], 'r', label='roll, tracked')
plt.plot(-allTh.T[:, 1], color=[0, 1, 0], label='pitch, tracked')
plt.plot(-allTh.T[:, 2], 'b', label='yaw, tracked')
plt.legend()
plt.grid()
plt.xlabel('Frame')
plt.ylabel('Angle [°]')
plt.show()