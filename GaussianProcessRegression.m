classdef GaussianProcessRegression < handle
    %GAUSSIANPROCESSREGRESSION Summary of this class goes here
    %   Detailed explanation goes here
    
    %% immutable properties
    properties (SetAccess = immutable, GetAccess = public)
        % selection of the kernel function
        kernelselection

        % class for kernel function
        kernel

        % training data
        xtrain % input
        ytrain % output
        datanum % numbers of training data
    end

    %% private properties
    properties (SetAccess = private, GetAccess = public)
        % kernel matrix
        KernelMat
        % inv(K) * y
        KernelInvY
    end

    %% class constant
    properties (Constant)
        % criteria to finish optimization of the hyper parameter
        diff_likelihood_criterion = 1;
    end

    %% Public methods
    methods (Access = public)
        %% Constructor
        function obj = GaussianProcessRegression(trainingdata, kernelselection, initdesignparams)
            %GAUSSIANPROCESSREGRESSION Construct an instance of this class

            obj.xtrain = trainingdata.input;
            obj.ytrain = trainingdata.output;
            obj.datanum = numel(trainingdata.output);

            obj.kernelselection = kernelselection;
            switch kernelselection
                case 'gaussian'
                    obj.kernel = GaussianKernel(initdesignparams);

                case 'periodic'
                    obj.kernel = PeriodicKernel(initdesignparams);

                otherwise
                    error('No maching kernel selection')
            end
        end
        
        %% General features
        function likelihood = regression(obj)

            % calculate the kernel matrix
            obj.calcKernelMat();

            % return the likelihood of the regression
            likelihood = obj.calc_likelihood();
        end

        function [mu, var] = predict(obj, xtest)

            N = obj.datanum;
            M = size(xtest, 2);

            % expected value and variance based on the regression
            mu = zeros(1, M);
            var = zeros(1, M);

            % $k_*$ function
            k = zeros(N, 1);

            % pre-calcuation
            yy = obj.KernelMat \ obj.ytrain';            
            for m = 1:M
                for n = 1:N
                    k(n, 1) = obj.kernel.kernel(obj.xtrain(:,n), xtest(:,m), n, N+m);
                end
            
                % $k_{**}$ function
                s = obj.kernel.kernel(xtest(:,m), xtest(:,m), m, m);
            
                % calculate the expected value and variance
                mu(1, m) = k' * yy;
                var(1, m) = s - k' / obj.KernelMat * k;
            end
            
            if any(var < 0) == true
                warning('Variance is negative!!! ')
                var(var<0) = 0;
            end
        end

        function fig = plotPrediction(obj, x_prediction, y_expected, y_variance)

            sigma = sqrt(y_variance);
            y_conf = [
                y_expected + 3 .* sigma;
                y_expected - 3 .* sigma
            ];
            
            x_conf_plot = [x_prediction x_prediction(end:-1:1)] ;         
            y_conf_plot = [y_conf(1, :), y_conf(2, end:-1:1)];
            
            figure
            hold on
            
            confrange = fill(x_conf_plot, y_conf_plot, 'red', 'DisplayName', '$ \pm 3 \sigma$');
            confrange.FaceColor = [255, 75, 0]./255;      
            confrange.EdgeColor = 'none';
            alpha(confrange, 0.3)

            regressionplot = plot(x_prediction, y_expected, '--', 'DisplayName', 'Regression');
            traindataplot = scatter(obj.xtrain, obj.ytrain, 'DisplayName', 'Data');
            
            legend([traindataplot, regressionplot, confrange], 'Interpreter', 'latex')
            xlabel('Input $x$', 'Interpreter', 'latex')
            ylabel('Output $y$', 'Interpreter', 'latex')

            fig = gcf();
        end

        %% Training process (Optimization of hyperparameters)
        function optparams = training(obj, trainingoption)

            if trainingoption.isdisplay, disp('GPR training start!'), end

            % apply initial value
            obj.kernel.updatehyperparams(trainingoption.initdesignparams)
            obj.calcKernelMat();

            previousparams = trainingoption.initdesignparams;
            
            % training the hyper parameter with the gradient descent method
            % note that this is maximization problem!
            previouslikelihood = -inf;
            lk_grad = zeros(3,1);
            likelihood_array = zeros(1, trainingoption.itermax);
            for idx = 1:trainingoption.itermax
                % update design parameters using the gradient
                if size(trainingoption.learningrate, 2) == 1
                    nextparams = previousparams + trainingoption.learningrate * lk_grad';
                else
                    nextparams = previousparams + trainingoption.learningrate * diag(lk_grad);
                end

                % update hyper parameter and kernel matrix
                obj.kernel.updatehyperparams(nextparams)
                obj.calcKernelMat();

                % calculate the likelihood and its gradient
                currentlikelihood = obj.calc_likelihood();
                lk_grad = obj.calc_likelihood_gradient();

                % for debug purpose
                disp([num2str(idx), '/', num2str(trainingoption.itermax), ' ', 'likelihood = ', num2str(currentlikelihood)])
                
                % store likelihood
                likelihood_array(1,idx) = currentlikelihood;

                % check the stop criteria
                diff_likelihood = currentlikelihood - previouslikelihood;
                if abs(diff_likelihood) < obj.diff_likelihood_criterion
                    if trainingoption.isdisplay, disp('GPR training completed (likelihood converged)'), end
                    optparams = nextparams;
                    break
                elseif isinf(currentlikelihood) && currentlikelihood < 0
                    if trainingoption.isdisplay, disp('GPR training completed (likelihood goes to -Inf)'), end
                    optparams = previousparams;
                    break
                elseif idx == trainingoption.itermax
                    if trainingoption.isdisplay, disp('GPR training completed (reached maximum iteration)'), end
                    optparams = nextparams;
                    break
                elseif isinf(currentlikelihood) && currentlikelihood > 0
                    if trainingoption.isdisplay, disp('GPR training terminated (likelihood goes to +Inf)'), end
                    optparams = previousparams;
                    break
                end

                % next iteration
                previousparams = nextparams;
                previouslikelihood = currentlikelihood;
            end

            % update GPR instance with the optimized design parameter
            obj.kernel.updatehyperparams(optparams)
            obj.calcKernelMat();

            if trainingoption.isdisplay
                figure
                plot(likelihood_array)
                xlabel('Iteration (step)')
                ylabel('Likelihood')
                title('Iterative process of the likelihood')
            end
        end

        function likelihood = calc_likelihood(obj)
            % calculate the likelihood of the 

            likelihood = -log(det(obj.KernelMat)) - obj.ytrain * obj.KernelInvY;
        end

        function likelihood_grad = calc_likelihood_gradient(obj)
            % calculate the gradient of the likelihood function
            grad1 = obj.calc_likelihood_grad1();
            grad2 = obj.calc_likelihood_grad2();
            grad3 = obj.calc_likelihood_grad3();
            
            likelihood_grad = [grad1; grad2; grad3];
        end
    end

    %% Private methods
    methods (Access = private)
        %% Calculation of the kernel matrix
        function calcKernelMat(obj)
            N = obj.datanum;
            K = zeros(N, N);
            for idx1 = 1:N
                for idx2 = 1:N
                    K(idx1, idx2) = obj.kernel.kernel(obj.xtrain(:,idx1), obj.xtrain(:,idx2), idx1, idx2);
                end
            end

            % check if the kernel matrix is positive semi-definite
            if any(eig(K) < 0)
                warning('Kernel matrix is not positive semi-definite!')
            end

            obj.KernelMat = K;
            obj.KernelInvY = K \ obj.ytrain';
        end

        %% gradient of the likelihood function
        function likelihood_grad1 = calc_likelihood_grad1(obj)
            N = obj.datanum;
            K_grad = zeros(N, N);
            for idx1 = 1:N
                for idx2 = 1:N
                    K_grad(idx1, idx2) = obj.kernel.grad1(obj.xtrain(:,idx1), obj.xtrain(:,idx2), idx1, idx2);
                end
            end
            likelihood_grad1 = -trace(obj.KernelMat \ K_grad) + obj.KernelInvY' * K_grad * obj.KernelInvY;
        end

        function likelihood_grad2 = calc_likelihood_grad2(obj)
            N = obj.datanum;
            K_grad = zeros(N, N);
            for idx1 = 1:N
                for idx2 = 1:N
                    K_grad(idx1, idx2) = obj.kernel.grad2(obj.xtrain(:,idx1), obj.xtrain(:,idx2), idx1, idx2);
                end
            end
            likelihood_grad2 = -trace(obj.KernelMat \ K_grad) + obj.KernelInvY' * K_grad * obj.KernelInvY;
                end

        function likelihood_grad3 = calc_likelihood_grad3(obj)
            N = obj.datanum;
            K_grad = zeros(N, N);
            for idx1 = 1:N
                for idx2 = 1:N
                    K_grad(idx1, idx2) = obj.kernel.grad3(obj.xtrain(:,idx1), obj.xtrain(:,idx2), idx1, idx2);
                end
            end
            likelihood_grad3 = -trace(obj.KernelMat \ K_grad) + obj.KernelInvY' * K_grad * obj.KernelInvY;
        end
    end
end
