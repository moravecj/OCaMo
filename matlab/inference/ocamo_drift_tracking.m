addpath('../include')

rng(42)
% Cyclicly go over 1500 frames.
sequenceLen = 1500;
% Cumulative +/-0.0005 rad decalibration in rotational parameters. 
dec = zeros(6, sequenceLen);
dec(1:3, :, :) = 2 * ((rand(3, sequenceLen) > 0.5) - 0.5) * 0.0005;
dec = cumsum(dec, 2);
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
% SGD Schaul tracking initialization wiuthout boundary
calibrationAlgorithm = SGDSchaul([0.001;0.001;0.001], 3, 1, [0;0;0], 1);
% Directory with pointclouds and images
data_dir = '../../example/carla_example/';
if ~exist(data_dir, 'dir')
    error('Please download and extract the example dataset, at: https://cmp.felk.cvut.cz/~moravj34/data/carla_example.zip.')
end

allThetas = zeros(sequenceLen, 3);
% Go over 1500 frames
i = 0;
while i < sequenceLen
    % Load image and pointcloud
    img = imread(sprintf('%s/%s.jpg', data_dir, sprintf('%03d',mod(i, 200) + 1)));
    load(sprintf('%s/%s.mat', data_dir, sprintf('%03d',mod(i, 200) + 1))', 'points');
    % Extract pointcloud corners and image edges
    [corners] = LiDARProcessor.findWaymoCorners(points', opt);
    [~, edges] = edge_canny(rgb2gray(img), 1);
    KDTree = createns(edges,'nsmethod','kdtree');
    % Decalibrate
    corners_dec = LiDARProcessor.transormPoints(corners, dec(1:3, i + 1), dec(4:6, i + 1));
    % Track
    theta = calibrationAlgorithm.processNewFrame(corners_dec, KDTree, cameraProcessor, opt);
    % Recalibrate (for visualization only)
    corners_rec = LiDARProcessor.transormPoints(corners_dec, theta(1:3), zeros(3, 1));
    
    allThetas(i + 1, :) = theta;
    i = i + 1;
end

%% VISUALIZATION

f = figure(1);
f.Position = [10 10 1920 1080];
ax1 = axes('Parent',f,'Units','normalized','Position',[0 0 1 1]);
ax2 = axes('Parent',f,'Units','normalized','OuterPosition',[0 0.7 1 .3]);
img(1:325, 150:end - 110, :) = 255;
image(ax1, img)
hold(ax1, 'on');
plot(ax1, [-100, -100],  [-100, -101], 'g', 'LineWidth',5)
plot(ax1, -100,  -100, 'b.', 'MarkerSize',60)
plot(ax1, -100,  -100, 'r.', 'MarkerSize',60);
[pcl2d, ~] = cameraProcessor.projectPointCloudsOriginalDistortion(corners_dec, eye(4));
[pcl2d_rec, ~] = cameraProcessor.projectPointCloudsOriginalDistortion(corners_rec, eye(4));
plot(ax1, pcl2d(1, :), pcl2d(2, :), 'b.', 'MarkerSize',25)
plot(ax1, pcl2d_rec(1, :), pcl2d_rec(2, :), 'r.', 'MarkerSize',25)
h = plot(ax1, edges(:, 1), edges(:, 2), 'g.', 'MarkerSize',2);
hold(ax1, 'off');

plot(ax2, 1:i, 180 / pi * dec(1, 1:i)', 'k', 'LineWidth', 2)
hold(ax2, 'on');
plot(ax2,1:i, 180 / pi * -squeeze(allThetas(1:i, 1)), 'r-', 'LineWidth', 3);
plot(ax2,1:i, 180 / pi * dec(2, 1:i)', 'm', 'LineWidth', 2)
plot(ax2,1:i, 180 / pi * -squeeze(allThetas(1:i, 2)), 'g-', 'LineWidth', 3);
plot(ax2,1:i, 180 / pi * dec(3, 1:i)', 'Color', [255,140,0] / 255, 'LineWidth', 2)
plot(ax2,1:i, 180 / pi * -squeeze(allThetas(1:i, 3)), 'b-', 'LineWidth', 3); hold off;
ylim(ax2,[-1, 2])
xlim(ax2,[0 1500])
ylabel(ax2, 'Angle [°]')
xlabel(ax2, 'Frame')
set(ax2, 'FontSize', 17)
grid(ax2, 'on');
hold(ax2, 'off');

lgd = legend(ax1, 'Image edges', ...
            'LiDAR corners with synthetic decalibration', ...
            'LiDAR corners after tracked recalibration', ...
            'Location', 'SouthWest');
        lgd.FontSize = 30;
        lgd.Position(3) = 0.5;

lgd = legend(ax2,'roll, decalibration', 'roll, tracked', ...
'pitch, decalibration', 'pitch, tracked', ...
'yaw, decalibration', 'yaw, tracked', 'Location', 'SouthWest', 'FontSize', 11);
lgd.Position(2) = 0.89;
lgd.Position(1) = 0.15;
lgd.Position(3) = 0.13;
lgd.FontSize = 12;

axis(ax1, 'off')