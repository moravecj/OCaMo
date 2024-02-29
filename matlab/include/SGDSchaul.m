classdef SGDSchaul < handle
    properties(Access = private)
        theta_
        dTheta_
        
        batchCount_
        batchId_
        currentCountInBatch_
        
        gradients_
        hessians_
        gradientsSquared_
        
        tau_
        meanGradients_
        meanHessians_
        meanGradientsSquared_
        
        burnIn_
        
        bound_
    end
    properties(Access = public)
        fVal = 0
        
        allGrads = []
    end
    methods
        function obj = SGDSchaul(dTheta, countOfParameters, batchCount, intialTheta, ~)
            assert(batchCount > 0 & floor(batchCount) == batchCount, 'Batch count has to be possitive integer.');
            obj.dTheta_ = dTheta(1:countOfParameters);
            if nargin < 4
                obj.theta_ = zeros(countOfParameters, 1);
            else
                obj.theta_ = intialTheta;
            end
            obj.bound_ = 1;
            if nargin == 5
                obj.bound_ = 0;
            end
            obj.tau_ = 1 * ones(countOfParameters, 1);
            obj.batchId_ = 0;
            obj.batchCount_ = batchCount;
            
            obj.meanGradients_ = zeros(countOfParameters, 1);
            obj.meanHessians_ = zeros(countOfParameters, 1);
            obj.meanGradientsSquared_ = zeros(countOfParameters, 1);
            
            obj.currentCountInBatch_ = 0;
            obj.gradients_ = zeros(countOfParameters, 1);
            obj.gradientsSquared_ = zeros(countOfParameters, 1);
            obj.hessians_ = zeros(countOfParameters, 1);
            
            obj.burnIn_ = 10;
        end
        
        function theta = processNewFrame(obj, corners, KDTree, cameraProcessor, opt)
            [fTheta,~] = likelihoodEstimationKernelCorrelation(obj.theta_, corners, KDTree, cameraProcessor, opt);
            fThetaPlusDtheta = zeros(length(obj.theta_), 1);
            fThetaMinusDtheta = zeros(length(obj.theta_), 1);
            
            for i = 1:length(obj.theta_)
                theta = obj.theta_;
                theta(i) = theta(i) + obj.dTheta_(i);
                fThetaPlusDtheta(i) = likelihoodEstimationKernelCorrelation(theta, corners, KDTree, cameraProcessor, opt);
            end
            
            for i = 1:length(obj.theta_)
                theta = obj.theta_;
                theta(i) = theta(i) - obj.dTheta_(i);
                fThetaMinusDtheta(i) = likelihoodEstimationKernelCorrelation(theta, corners, KDTree, cameraProcessor, opt);
            end
            obj.allGrads(end + 1, :) = abs((fThetaPlusDtheta - 2 * fTheta + fThetaMinusDtheta) ./ (obj.dTheta_.^2));
            obj.currentCountInBatch_ = obj.currentCountInBatch_ + 1;
            % Accumulate numerical gradients.
            g = (fThetaPlusDtheta - fThetaMinusDtheta) ./ (2 * obj.dTheta_);
            obj.gradients_ = obj.gradients_ + g;
            % Accumulate numerical gradients squared.
            obj.gradientsSquared_ = obj.gradientsSquared_  + g .^ 2;
            % Accumulate numerical hessians.
            obj.hessians_ = obj.hessians_ + abs((fThetaPlusDtheta - 2 * fTheta + fThetaMinusDtheta) ./ (obj.dTheta_.^2));
                              
            if obj.currentCountInBatch_ >= obj.batchCount_ && obj.burnIn_ < 1
                % Calculate gradients, squared gradients, hessians.
                obj.gradientsSquared_ = obj.gradientsSquared_ / obj.currentCountInBatch_;
                obj.gradients_ = obj.gradients_ / obj.currentCountInBatch_;
                obj.hessians_ = obj.hessians_ / obj.currentCountInBatch_;
                
                % Filter gradients, squared gradients and hessians.
                obj.meanGradients_ = (1 - 1 ./ obj.tau_) .* obj.meanGradients_ + (1 ./ obj.tau_) .* obj.gradients_;
                obj.meanHessians_ = (1 - 1 ./ obj.tau_) .* obj.meanHessians_ + (1 ./ obj.tau_) .* obj.hessians_;
                obj.meanGradientsSquared_ = (1 - 1 ./ obj.tau_) .* obj.meanGradientsSquared_ + ...
                                            (1 ./ obj.tau_) .* obj.gradientsSquared_;
                % Estimate learning rate.
                learningRate = (obj.meanGradients_.^2) ./ (obj.meanGradientsSquared_ + eps);
                learningRate(isinf(learningRate)) = 0;
                % Estimate next memory size.
                obj.tau_ = (1 - (obj.meanGradients_.^2) ./ (obj.meanGradientsSquared_ + eps)) .* obj.tau_ + 1;
                obj.tau_(obj.tau_ > 5) = 5;
                
                nu = obj.gradients_ ./ obj.meanHessians_;
                nu(1:3) = min(abs(nu(1:3)), 0.0024) .* nu(1:3) ./ (abs(nu(1:3)) + eps);
                % SGD step
                obj.theta_ = obj.theta_ - learningRate .* nu;
                if obj.bound_
                    if(abs(obj.theta_(1)) > 0.0166)
                        obj.theta_(1) = 0.0166 * abs(obj.theta_(1)) ./ obj.theta_(1);
                    end
                    if(abs(obj.theta_(2)) > 0.0083)
                        obj.theta_(2) = 0.0083 * abs(obj.theta_(2)) ./ obj.theta_(2);
                    end
                    if(abs(obj.theta_(3)) > 0.0026)
                        obj.theta_(3) = 0.0026 * abs(obj.theta_(3)) ./ obj.theta_(3);
                    end
                end
                
                obj.gradients_ = zeros(length(obj.theta_), 1);
                obj.gradientsSquared_ = zeros(length(obj.theta_), 1);
                obj.hessians_ = zeros(length(obj.theta_), 1);
                
                obj.currentCountInBatch_ = 0;
            end
            obj.batchId_ = obj.batchId_ + 1;
            obj.burnIn_ = obj.burnIn_ - 1;
            theta = obj.theta_;
        end
        function b = getBatchId(obj)
            b = obj.batchId_;
        end
    end
end

