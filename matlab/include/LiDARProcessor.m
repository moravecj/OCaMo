classdef LiDARProcessor
    methods(Static)
       function [corners, idx] = findWaymoCorners(velo, opt)
            % Change in distance. 
            r = sqrt(sum(velo(1:3, :).^2, 1));
            d_t_m = conv(r, opt.mask, 'same');
            d_2 = conv(r.^2, ones(1, size(opt.mask, 2)), 'same');
            c = d_t_m ./ sqrt(d_2);
            
            az = atan2(velo(2, :),  velo(1, :));
            
            dtheta = abs(diff(az)) < 0.3;
            dtheta(1:end - 1) = dtheta(2:end) .* dtheta(1:end-1);
            dtheta(2:end) = dtheta(2:end) .* dtheta(1:end-1);
            
            % Choose closer.
            r_right = [2:length(c) length(c)];
            r_left = [1 1:length(r)-1];
            r_right_idx = r(r_right) < r - 0.5;
            r_left_idx = r(r_left) < r - 0.5;
            r_closest = 1:length(r);
            r_closest(r_right_idx) = r_right(r_right_idx);
            r_closest(r_left_idx) = r_left(r_left_idx);
            
            % non-maxima suppresion for change in distance
            ok = nonMaximaSuppression(c, opt.NMSLiDARDistance, opt.TLiDARDistance); 
            
            % Change in intensity.
            % CARLA SIMULATOR DOES NOT PROVIDE LIDAR INTENSITY.
            % r = velo(4, :);
            % d = conv(r, opt.mask, 'same');
            % ok1 = nonMaximaSuppression(d, opt.NMSLiDARIntensity, opt.TLiDARIntensity);
            
            % Select corners.
            idx = false( size( c ) );
            idx(r_closest([dtheta 1] & ok)) = 1;
            idx([dtheta 1] & [abs(diff(az)) 0] > opt.TLiDARAzimuth) = 1;
            idx([dtheta 1] & [0 abs(diff(az))] > opt.TLiDARAzimuth) = 1;
            %idx(r_closest([dtheta 1] & ok1)) = 1;
            
            corners = velo(1:3, idx);
       end
       function pcl = transormPoints(pcl, omega, t)
           if nargin < 3
               t = zeros(3, 1);
           end
           % Exponential map for rotation.
           logT = [asym(omega(1:3)), zeros(3, 1); 0 0 0 0];
           T = explogRT3(logT);
           % Add translation.
           T(1:3, 4) = t;
           % Transform points.
           pcl = T * [pcl(1:3,:); ones(1, size(pcl, 2))];
           pcl = pcl(1:3, :);
       end
   end
end

