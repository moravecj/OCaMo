addpath('../include')

rng(42)
% Decalibration from <-0.02, -0.01> ∪ <0.01, 0.02> rad in rotation and 
% <-0.2, -0.1> ∪ <0.1, 0.2> m in translation.
dec = zeros(6, 1);
dec(1:3) = 2 * ((rand(3, 1) > 0.5) - 0.5) .* (rand(3, 1) * 0.01 + 0.01);
dec(4:6) = 2 * ((rand(3, 1) > 0.5) - 0.5) .* (rand(3, 1) * 0.1 + 0.1);
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
% SGD Schaul tracking initialization for calibrated and decalibrated scenarios.
calibrationAlgorithmCalib = SGDSchaul([0.001;0.001;0.001], 3, 1, [0;0;0]);
calibrationAlgorithmDecalib = SGDSchaul([0.001;0.001;0.001], 3, 1, [0;0;0]);
% Directory with pointclouds and images
data_dir = '../../example/carla_example/';
if ~exist(data_dir, 'dir')
    error('Please download and extract the example dataset, at: https://cmp.felk.cvut.cz/~moravj34/data/carla_example.zip.')
end

allThetas = zeros(200, 3);
allThetasDec = zeros(200, 3);
for i = 1:200
    % Load image and pointcloud
    img = imread(sprintf('%s/%s.jpg', data_dir, sprintf('%03d',i)));
    load(sprintf('%s/%s.mat', data_dir, sprintf('%03d',i))', 'points');
    % Extract pointcloud corners and image edges
    [corners] = LiDARProcessor.findWaymoCorners(points', opt);
    [~, edges] = edge_canny(rgb2gray(img), 1);
    KDTree = createns(edges,'nsmethod','kdtree');
    % Track calibrated scenario
    theta = calibrationAlgorithmCalib.processNewFrame(corners, KDTree, cameraProcessor, opt);
    allThetas(i, :) = theta;
    % Synthetically decalibrated conrners between frame 51 and 110
    if i > 50 && i < 111
        corners = LiDARProcessor.transormPoints(corners, dec(1:3), dec(4:6));
    end
    % Track synthetically decalibrated scenario
    theta = calibrationAlgorithmDecalib.processNewFrame(corners, KDTree, cameraProcessor, opt);
    allThetasDec(i, :) = theta;
end

%% VISUALIZATION

omega_est = [0.0166 0.0083 0.0026] / 5;
theta_thr = 3 * omega_est;
theta_bnd = 5 * omega_est;

p_j = 1 - ((1 / 2) * erfc((theta_thr - allThetas(:, 1:3)) ./ (sqrt(2) * omega_est)) + (1 / 2) * erfc((theta_thr + allThetas(:, 1:3)) ./ (sqrt(2) * omega_est)));
p1 = p_j(:, 1) .* p_j(:, 2) .* p_j(:, 3);
p_j = 1 - ((1 / 2) * erfc((theta_thr - allThetasDec(:, 1:3)) ./ (sqrt(2) * omega_est)) + (1 / 2) * erfc((theta_thr + allThetasDec(:, 1:3)) ./ (sqrt(2) * omega_est)));
p2 = p_j(:, 1) .* p_j(:, 2) .* p_j(:, 3);

figure(1)
clf;
plot(p1, 'LineWidth', 2)
hold on;
plot(p2, 'LineWidth', 2)
hold off;
ylabel('$$V_\mathrm{OC}$$', 'Interpreter','latex', 'FontSize',15)
xlabel('Frame', 'Interpreter','latex', 'FontSize',15)
title('OCaMo Calibration Monitoring')
legend('Calibrated', 'Decalibrated', 'Location','southeast', 'fontsize', 13)
grid on;