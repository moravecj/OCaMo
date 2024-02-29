from matplotlib import pyplot as plt
import cv2
from scipy.io import loadmat
import numpy as np
from scipy import spatial
from scipy.spatial.transform import Rotation as R
from scipy.stats import beta

from include.feature_extraction import get_corners
from include.kernel_correlation import estimate_kernel_correlation

np.random.seed(42)

# evaluate f-index on ROTATIONAL PARAMETERS ONLY!
# the original approach uses all extrinsic parameters
# see matlab results for more precise evaluation
def evaluate_f_index(corners, edges_kdtree):
    f_index = np.zeros((27,))
    cnt = 0
    # iterate over all 3^3 combinations of parameters
    for rx in [-0.01, 0, 0.01]:
        for ry in [-0.01, 0, 0.01]:
            for rz in [-0.01, 0, 0.01]:
                param = np.array([rx, ry, rz])
                # estimate kernel correlation on these parameters
                f_index[cnt] = estimate_kernel_correlation(edges_kdtree, corners, param, 9, 10)
                cnt += 1
    return f_index

data_dir = '../../example/carla_example'
T_lid2cam = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[(1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 0, 1920 / 2], [0, (1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 1080 / 2], [0, 0, 1]])

# synthetic decalibration from <-0.02, -0.01> ∪ <0.01, 0.02> rad in rotations and <-0.2, -0.1> ∪ <0.1, 0.2> m in translations
dec = np.zeros((6,))
dec[:3] = (2 * (np.random.rand(3,) > 0.5) - 1) * (np.random.rand(3,) * 0.01 + 0.01)
dec[3:] = (2 * (np.random.rand(3,) > 0.5) - 1) * (np.random.rand(3,) * 0.1 + 0.1)

T_dec = np.vstack((np.hstack((R.from_rotvec(dec[:3], degrees=False).as_matrix(), dec[3:6, None])), np.array([0, 0, 0, 1])))

allTh = np.zeros((27, 200))
allTh_dec = np.zeros((27, 200))

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
    # evaluate f-index on reference paramters
    allTh[:, i] = evaluate_f_index(cor, kdtree)
    # decalibrate between frames 51 and 110
    cor_dec = cor.copy()
    if i > 50 and i < 110:
        cor_dec = (T_dec[:3, :3] @ cor_dec.T + T_dec[:3, 3:]).T
    # evaluate f-index on decalibrated parameters
    allTh_dec[:, i] = evaluate_f_index(cor_dec, kdtree)

# beta distribution parameters for V-index evaluation
# modified due to the use of rotational parameters for monitoring only, in this python version
a_c, b_c = 10, 0.4
a_d, b_d = 4.08, 3.70
# estimate the V-index with beta distributions, see (15)
# calibrated
pc = beta.pdf(np.mean(allTh[13, :] <= allTh[:, :], axis=0) - 10**-16, a=a_c, b=b_c)
pd = beta.pdf(np.mean(allTh[13, :] <= allTh[:, :], axis=0) - 10**-16, a=a_d, b=b_d)
v_lto1 = pc / (pc + pd)
# decalibrated
pc = beta.pdf(np.mean(allTh_dec[13, :] <= allTh_dec[:, :], axis=0) - 10**-16, a=a_c, b=b_c)
pd = beta.pdf(np.mean(allTh_dec[13 , :] <= allTh_dec[:, :], axis=0) - 10**-16, a=a_d, b=b_d)
v_lto2 = pc / (pc + pd)
# 9-frames window
pom = np.zeros((200,))
pom[:9] = 1
plt.plot(np.convolve(v_lto1, np.hstack((np.ones(9), np.zeros(9))), 'full')[:-17]/np.cumsum(pom), label='Calibrated')
plt.plot(np.convolve(v_lto2, np.hstack((np.ones(9), np.zeros(9))), 'full')[:-17]/np.cumsum(pom), label='Decalibrated')
plt.grid()
plt.legend()
plt.xlabel('Frame')
plt.ylabel('LTO Validity Index')
plt.show()