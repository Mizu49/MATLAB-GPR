%% Parametric study of the inital design parameters
close all
clear

%% Data
N_sample = 200;

x_data = [
    linspace(0, 10, N_sample);
];

y_nominal = @(x) sin(5*x) + sin(3*x);

a = 2e-1;
b = 0;
noise = a.*randn(1, N_sample) + b;
y_data = y_nominal(x_data) + noise;

dim_x = size(x_data, 1);

trainingdata = struct('input', x_data, 'output', y_data);

%% Parameters 

theta_1 = -5:1:5;
theta_2 = -5:1:5;
theta_3 = -5:1:1;

parameters = combvec(theta_1, theta_2, theta_3);
paramnum = size(parameters, 2);

%% Parametric study with gaussian process

kernelselection = 'gaussian';

array_likelihood = zeros(1, paramnum);
parfor idx = 1:paramnum
    gpr = GaussianProcessRegression(trainingdata, kernelselection, parameters(:, idx)');
    array_likelihood(1, idx) = gpr.regression();
end

% Post process
if any(isinf(array_likelihood))
    warning('Likelihood has infinity value, replaced by `-inf`')
    array_likelihood(isinf(array_likelihood)) = -inf;
end
[best_likelihood, best_params] = max(array_likelihood);
disp(['Best likelihood is ', num2str(best_likelihood), ' with params = '])
disp(parameters(:, best_params)')

%% Gaussian process regression

gpr = GaussianProcessRegression(trainingdata, kernelselection, parameters(:, best_params)');
gpr.regression();

[y_mu, y_variance] = gpr.predict(x_data);

sigma = sqrt(y_variance);
y_conf = [
    y_mu + 3 .* sigma;
    y_mu - 3 .* sigma
];
N_sample = numel(y_mu);
x_conf_plot = [x_data, x_data(end:-1:1)] ;         
y_conf_plot = [y_conf(1, :), y_conf(2, end:-1:1)];

figure;
hold on;

confrange = fill(x_conf_plot, y_conf_plot, 'red', 'DisplayName', '$ \pm 3 \sigma$');
confrange.FaceColor = [255, 75, 0]./255;      
confrange.EdgeColor = 'none';
alpha(confrange, 0.3)

trainplot = scatter(trainingdata.input, trainingdata.output, 'DisplayName', 'Training data');
regplot = plot(x_data, y_mu, 'DisplayName', 'Regression');
xlabel('Input')
ylabel('Output')
legend([trainplot, regplot, confrange], 'Interpreter', 'latex')