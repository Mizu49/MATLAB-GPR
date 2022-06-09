classdef PeriodicKernel < handle
    %GAUSSIANKERNEL
    
    %% Properties
    properties (SetAccess = private, GetAccess = public)
        % design parameter for the hyper parameter
        designparams
        % hyper parameter
        hyperparams

    end

    properties (Constant)
        
        % dimension of the hyper parameter
        dimhyperparams = 3;
    end
    
    %% Public properties
    methods (Access = public)
        function obj = PeriodicKernel(initdesignparams)
            %GAUSSIANKERNEL Construct an instance of this class

            % apply the design parameter
            obj.designparams = initdesignparams;
            % update the hyperparameter
            obj.update_hyperparams();
        end

        function updatehyperparams(obj, newparams)
            obj.designparams = newparams;
            obj.update_hyperparams();
        end
        
        function kernelvalue = kernel(obj, x1, x2, idx1, idx2)

            p1 = obj.hyperparams(1,1);
            p2 = obj.hyperparams(1,2);
            p3 = obj.hyperparams(1,3);
            
            kernelvalue = p1 * exp(cos(abs(x1 - x2))/p2) + p3 * deltafun(idx1, idx2);
        end

        function grad = grad1(obj, x1, x2, idx1, idx2)
            % hyper parameters
            p1 = obj.hyperparams(1,1);
            p2 = obj.hyperparams(1,2);
            p3 = obj.hyperparams(1,3);            
            
            grad = p1 * exp( cos(abs(x1 - x2))/p2 );
        end

        function grad = grad2(obj, x1, x2, idx1, idx2)
            % hyper parameters
            p1 = obj.hyperparams(1,1);
            p2 = obj.hyperparams(1,2);
            p3 = obj.hyperparams(1,3);

            grad = - p1 * exp(cos(abs(x1 - x2)/p2)) * cos(abs(x1 - x2))/p2;
        end

        function grad = grad3(obj, x1, x2, idx1, idx2)
            grad = deltafun(idx1, idx2);
        end
    end

    %% Private properties
    methods (Access = private)
        function update_hyperparams(obj)
            % update the hyper parameters based on the design parameters
            obj.hyperparams = exp(obj.designparams);
        end
    end
end

% delta function for the two indices
function delta = deltafun(idx1, idx2)
    if idx1 == idx2
        delta = 1;
    else
        delta = 0;
    end
end

