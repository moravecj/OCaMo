from matplotlib import pyplot as plt
import cv2
from scipy.io import loadmat
import numpy as np
from scipy import spatial
from scipy import special
from scipy.spatial.transform import Rotation as R

from include.feature_extraction import get_corners
from include.SGDSchaul import SGDSchaul

np.random.seed(42)

data_dir = '../../example/carla_example'
T_lid2cam = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[(1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 0, 1920 / 2], [0, (1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 1080 / 2], [0, 0, 1]])

# synthetic decalibration from <-0.02, -0.01> ∪ <0.01, 0.02> rad in rotations and <-0.2, -0.1> ∪ <0.1, 0.2> m in translations
dec = np.zeros((6,))
dec[:3] = (2 * (np.random.rand(3,) > 0.5) - 1) * (np.random.rand(3,) * 0.01 + 0.01)
dec[3:] = (2 * (np.random.rand(3,) > 0.5) - 1) * (np.random.rand(3,) * 0.1 + 0.1)
T_dec = np.vstack((np.hstack((R.from_rotvec(dec[:3], degrees=False).as_matrix(), dec[3:6, None])), np.array([0, 0, 0, 1])))
# initialize two trackers for calibrated and decalibrated (between frames 50 a 110) sequence
sgd = SGDSchaul(np.array([0, 1, 2]), bnd=np.array([0.0166, 0.0083, 0.0026]))
sgd_dec = SGDSchaul(np.array([0, 1, 2]), bnd=np.array([0.0166, 0.0083, 0.0026]))

allTh = np.zeros((3, 200))
allTh_dec = np.zeros((3, 200))

# iterate over all frames
for i in range(200):
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
    cor_dec = cor.copy()
    # update the tracker on the calibrated sequence
    sgd.update(cor, kdtree)
    # decalibrate between frames 51 and 110
    if i > 50 and i < 110:
        cor_dec = (T_dec[:3, :3] @ cor_dec.T + T_dec[:3, 3:]).T
    # update the tracker on the decalibrated sequence
    sgd_dec.update(cor_dec, kdtree)

    allTh[:, i] = sgd.theta_.copy()
    allTh_dec[:, i] = sgd_dec.theta_.copy()
# evaluate V-index of both trackers
omega_est = np.array([0.0166, 0.0083, 0.0026])[:, None] / 5
theta_thr = 3 * omega_est
theta_bnd = 5 * omega_est
p_j1 = 1 - ((1 / 2) * special.erfc((theta_thr - allTh) / (np.sqrt(2) * omega_est)) + (1 / 2) * special.erfc((theta_thr + allTh) / (np.sqrt(2) * omega_est)))
p1 = p_j1[0] * p_j1[1] * p_j1[2]
p_j2 = 1 - ((1 / 2) * special.erfc((theta_thr - allTh_dec) / (np.sqrt(2) * omega_est)) + (1 / 2) * special.erfc((theta_thr + allTh_dec) / (np.sqrt(2) * omega_est)))
p2 = p_j2[0] * p_j2[1] * p_j2[2]

plt.plot(p1, label='Calibrated')
plt.plot(p2, label='Decalibrated')
plt.grid()
plt.legend()
plt.xlabel('Frame')
plt.ylabel('OCaMo Validity Index')
plt.show()