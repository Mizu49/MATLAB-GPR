function trainingoption = settrainingoptions(profile, kernelprofile)
%SETTRAININGOPTIONS Summary of this function goes here

switch profile
    case 'A'
        switch kernelprofile
            case 'gaussian'
                trainingoption.initdesignparams = [1, -1, -3];
                trainingoption.learningrate = 1e-3;
                trainingoption.itermax = 100;
                trainingoption.isdisplay = true;

            case 'periodic'
                trainingoption.initdesignparams = [-2, -3, -2];
                trainingoption.learningrate = 5e-5;
                trainingoption.itermax = 100;
                trainingoption.isdisplay = true;
            otherwise
                error('No matching `kernelprofile` found!!')
        end
    otherwise
        error('`profile` not found!')
end

end

