addpath('../include')

rng(42)
% Default parameters from WaymoA.
opt = initOptions();
% CARLA extrinsic and intrinsic calibration (based on the simulated setup).
T_lid2cam = [[0, -1, 0, 0]; [0, 0, -1, 0]; [1, 0, 0, 0]; [0 0 0 1]] * ... 
    [[1, 0, 0, 0]; [0, 1, 0, -0.2]; [0, 0, 1, 0]; [0, 0, 0, 1]];
K = [[(1920 * 0.5) / tan(0.5 * (60.0) * pi / 180.0), 0, 1920 / 2]; ...
    [0, (1920 * 0.5) / tan(0.5 * (60.0) * pi / 180.0), 1080 / 2]; ...
    [0, 0, 1]];
cameraProcessor = CameraProcessor(T_lid2cam, K, ...
                          [0,0,0,0,0], [1920, 1080]);
% Directory with pointclouds and images and one selected frame.
data_dir = '../../example/carla_example/';
frame_id = '100';
if ~exist(data_dir, 'dir')
    error('Please download and extract the example dataset, at: https://cmp.felk.cvut.cz/~moravj34/data/carla_example.zip.')
end
% Load image and pointcloud
img = imread(sprintf('%s/%s.jpg', data_dir, frame_id));
load(sprintf('%s/%s.mat', data_dir, frame_id)', 'points');
% Extract pointcloud corners and image edges
[corners] = LiDARProcessor.findWaymoCorners(points', opt);
[~, edges] = edge_canny(rgb2gray(img), 1);
% Projection of detected LiDAR corners on the image.
[pcl2d, ~] = cameraProcessor.projectPointCloudsOriginalDistortion(corners, eye(4));

f = figure(2);
clf;
f.Position = [10 10 1920 1080];
ax1 = axes('Parent',f,'Units','normalized','Position',[0 0 1 1]);
imshow(img)
hold on;
plot(edges(:, 1), edges(:, 2), 'g.', 'MarkerSize',2)
plot(pcl2d(1, :), pcl2d(2, :), 'rx', 'MarkerSize',8, 'LineWidth',3)
hold off;