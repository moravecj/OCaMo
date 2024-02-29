classdef CameraProcessor
    properties(Access = private)
        tLiDAR2Camera_
        cameraMatrix_
        distortionCoefficients_
        imageSize_
    end
    
    methods
        function obj = CameraProcessor(tLiDAR2Camera, cameraMatrix, distortionCoefficients, imageSize)
            % CameraProcessor prepares camera processor for projection and edge detection.
            obj.tLiDAR2Camera_ = tLiDAR2Camera;
            obj.cameraMatrix_ = cameraMatrix;
            obj.distortionCoefficients_ = distortionCoefficients;
            obj.imageSize_ = imageSize;
        end
        function [pcl2d, idx] = projectPointCloudsOriginalDistortion(obj, pcl, T)
            % projectPointCloudsOriginalDistortion projects pointcloud on an image plane using KITTI radial distortion.
            if nargin < 3
                T = eye(4);
            end
            K = obj.cameraMatrix_;
            % Transformation to the camera coordinate system.
            pcl2d = obj.tLiDAR2Camera_ * T * [pcl(1:3,:); ones(1, size(pcl, 2))];

            idx1 = pcl2d(3, :) > 1;
            pcl2d = pcl2d ./ pcl2d(3, :);
            pcl2d(:, ~idx1) = -Inf;

            % Check which points will be correctly projected on a plane, before applying radial distortion.
            pcl2dPom = K * pcl2d(1:3, :);
            idx2 = pcl2dPom(1, :) >= -200 & pcl2dPom(1, :) < obj.imageSize_(1) + 200 & pcl2dPom(2, :) >= -200 & pcl2dPom(2, :) < obj.imageSize_(2) + 200;
            pcl2d(:, ~idx2) = -Inf;
            
            % Apply radial and tangential distortions.
            r2 = pcl2d(1, :) .^ 2 + pcl2d(2, :) .^ 2;
            k1 = obj.distortionCoefficients_(1);
            k2 = obj.distortionCoefficients_(2);
            p1 = obj.distortionCoefficients_(3);
            p2 = obj.distortionCoefficients_(4);
            k3 = obj.distortionCoefficients_(5);
            
            x = pcl2d(1, :);
            y = pcl2d(2, :);
            
            pcl2d(1, :) = x .* (1 + k1 .* r2 + k2 .* (r2.^2) + k3 .* (r2 .^ 3)) + 2 .* p1 .* x .* y + p2 .* (r2 + 2 .* (x .^ 2));
            pcl2d(2, :) = y .* (1 + k1 .* r2 + k2 .* (r2.^2) + k3 .* (r2 .^ 3)) + 2 .* p2 .* x .* y + p1 .* (r2 + 2 .* (y .^ 2));
            
            pcl2d = K * pcl2d(1:3, :);
            
            pcl2d = pcl2d(1:2, :);
            % Cut points that are not projected to the image.
            idx3 = pcl2d(1, :) >= 0 & pcl2d(1, :) < obj.imageSize_(1) & pcl2d(2, :) >= 0 & pcl2d(2, :) < obj.imageSize_(2);
            idx = idx1 & idx2 & idx3;
            pcl2d = pcl2d(:, idx);
        end
    end
end

