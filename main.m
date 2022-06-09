%% Test script for Gaussian Process regression
close all
clear

%% Data
N_sample = 200;

x_data = [
    linspace(0, 10, N_sample);
];

y_nominal = @(x) sin(5*x) + sin(3*x);

a = 1e-1;
b = 0;
noise = a.*randn(1, N_sample) + b;
y_data = y_nominal(x_data) + noise;

dim_x = size(x_data, 1);

trainingdata = struct('input', x_data, 'output', y_data);

%% Gaussian process

profile = 'A';
kernelselection = 'gaussian';

% create gpr instance
designparams = setdesignparams(profile, kernelselection);
gpr = GaussianProcessRegression(trainingdata, kernelselection, designparams);

% training hyper parameter
trainingoption = settrainingoptions(profile, kernelselection);
gpr.training(trainingoption);

% update regression
likelihood = gpr.regression();

%% Test regression
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
