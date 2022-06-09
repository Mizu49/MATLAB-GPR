function hyperparams = setdesignparams(profile, kernelprofile)

switch profile
    case 'A'
        switch kernelprofile
            case 'gaussian'
                hyperparams = [1, -1, -3];

            case 'periodic'
                hyperparams = [0, -1, -3];
            otherwise
                error('No matching `kernelprofile` found!!')
        end
    otherwise
        error('`profile` not found!')
end

