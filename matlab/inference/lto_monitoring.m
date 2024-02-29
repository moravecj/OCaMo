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
% Directory with pointclouds and images
data_dir = '../../example/carla_example/';
if ~exist(data_dir, 'dir')
    error('Please download and extract the example dataset, at: https://cmp.felk.cvut.cz/~moravj34/data/carla_example.zip.')
end

allThetas = zeros(200, 729);
allThetasDec = zeros(200, 729);
for i = 1:200
    % Load image and pointcloud
    img = imread(sprintf('%s/%s.jpg', data_dir, sprintf('%03d',i)));
    load(sprintf('%s/%s.mat', data_dir, sprintf('%03d',i))', 'points');
    % Extract pointcloud corners and image edges
    [corners] = LiDARProcessor.findWaymoCorners(points', opt);
    [~, edges] = edge_canny(rgb2gray(img), 1);
    KDTree = createns(edges,'nsmethod','kdtree');
    % Evaluate F-index on calibrated sequence
    [proj, proj_idx] = project_corners(double(corners(1:3, :)'), double(T_lid2cam), double(K), 0.01, 0.01, 0.1, 0.1, 729);
    x_idx = round(proj_idx(1, sum(proj == -1, 1) == 0)');
    x = proj(1:2, sum(proj == -1, 1) == 0)';
    [~, dist] = knnsearch(KDTree,x,'k',opt.k);
    J = evaluate_J_ocamo(dist, x_idx, 729, opt.sigma, size(KDTree.X, 1));
    allThetas(i, :) = J;
    % Synthetically decalibrated conrners between frame 51 and 110
    corners_decalib = corners;
    if i > 50 && i < 111
        corners_decalib = LiDARProcessor.transormPoints(corners_decalib, dec(1:3), dec(4:6));
    end
    % Evaluate F-index on decalibrated sequence
    [proj, proj_idx] = project_corners(double(corners_decalib(1:3, :)'), double(T_lid2cam), double(K), 0.01, 0.01, 0.1, 0.1, 729);
    x_idx = round(proj_idx(1, sum(proj == -1, 1) == 0)');
    x = proj(1:2, sum(proj == -1, 1) == 0)';
    [~, dist] = knnsearch(KDTree,x,'k',opt.k);
    J_decalib = evaluate_J_ocamo(dist, x_idx, 729, opt.sigma, size(KDTree.X, 1));
    allThetasDec(i, :) = J_decalib;
end
%% VISUALIZATION
pred = mean(allThetas(:, 365) <= allThetas(:, :), 2)' - 10^-16;
pred_dec = mean(allThetasDec(:, 365) <= allThetasDec(:, :), 2)' - 10^-16;
p = movmean(pred, [9, 0]);
p = betapdf(p,40.5889,0.2032) ./ (betapdf(p,40.5889,0.2032) + betapdf(p, 4.0790, 3.6954));
p_dec = movmean(pred_dec, [9, 0]);
p_dec = betapdf(p_dec,40.5889,0.2032) ./ (betapdf(p_dec,40.5889,0.2032) + betapdf(p_dec, 4.0790, 3.6954));

figure(1)
clf;
plot(p, 'LineWidth', 2)
hold on;
plot(p_dec, 'LineWidth', 2)
hold off;
ylabel('$$V_\mathrm{LTO}$$', 'Interpreter','latex', 'FontSize',15)
xlabel('Frame', 'Interpreter','latex', 'FontSize',15)
title('LTO Calibration Monitoring')
legend('Calibrated', 'Decalibrated', 'Location','southeast', 'fontsize', 13)
grid on;