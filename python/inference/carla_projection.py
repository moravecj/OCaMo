from matplotlib import pyplot as plt
import cv2
from scipy.io import loadmat
import numpy as np

from include.feature_extraction import get_corners

data_dir = '../../example/carla_example/'
frame_id = 100
# reference extrinsic and intrinsic calibration
T_lid2cam = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
K = np.array([[(1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 0, 1920 / 2], [0, (1920 * 0.5) / np.tan(0.5 * (60.0) * np.pi / 180.0), 1080 / 2], [0, 0, 1]])
# load image and extract edges
img = plt.imread('{}/{}.jpg'.format(data_dir, str(frame_id).zfill(3)))
edges = cv2.Canny(img, 50, 100)
edg = np.where(edges)
edg = np.array((edg[0], edg[1]))
edg = edg[:, edg[0] > 1080 / 3]
# load pointcloud and extract corners
pcl = loadmat('{}/{}.mat'.format(data_dir, str(frame_id).zfill(3)))['points']
cor = get_corners(pcl)
# project corners on the image plate
T_pcl = T_lid2cam[:3, :3] @ cor[:, :3].T + T_lid2cam[:3, 3:]
proj = K @ T_pcl
proj = proj[:2, proj[2] > 0] / proj[2, proj[2] > 0] 
proj = proj[:,np.logical_and(np.logical_and(proj[1] >=0, proj[1] < 1080), np.logical_and(proj[0] >=0, proj[0] < 1920))]
# show projections and edges
plt.imshow(img)
plt.scatter(edg[1], edg[0], .1, 'g')
plt.scatter(proj[0], proj[1], 20, 'r', marker='x')
plt.xlim([0, 1920])
plt.ylim([1080, 0])
plt.show()