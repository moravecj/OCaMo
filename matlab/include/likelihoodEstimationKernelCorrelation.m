function [err, time] = likelihoodEstimationKernelCorrelation(theta, pcl, KDTree, cameraProcessor, opt)
    tic;
    % Apply rotation only.
    logT = [asym(theta(1:3)), zeros(3, 1); 0 0 0 0];
    T = explogRT3(logT);
    [X, ~] = cameraProcessor.projectPointCloudsOriginalDistortion(pcl, T);
    % Find nearest neighbours.
    [~, dist] = knnsearch(KDTree,X','k',opt.k);
    % Estimate likelihood.
    err = exp(-dist.^2  / (2*opt.sigma.^2));
    err = (-sum(sum(err, 2) /size(KDTree.X, 1))) / size(X, 2);
    time = toc;
end