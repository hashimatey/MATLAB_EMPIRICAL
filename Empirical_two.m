% Monte Carlo Experiment for Estimating Lambda in a Poisson Distribution
clear; clc;clear all


% Parameters
lambda_true = 2.5; % True population parameter
sample_sizes = [50, 500, 5000, 50000]; % Sample sizes
num_simulations = 1000; % Number of datasets per sample size

% Loop over sample sizes
for s = 1:length(sample_sizes)
    n = sample_sizes(s); % Current sample size
    fprintf('Running simulations for sample size n = %d...\n', n);
    
    % Initialize arrays to store estimates
    lambda_1 = zeros(num_simulations, 1); % MOM (1st moment)
    lambda_2 = zeros(num_simulations, 1); % MOM (2nd moment)
    lambda_3 = zeros(num_simulations, 1); % GMM (uniform)
    lambda_4 = zeros(num_simulations, 1); % GMM (two-step)
    lambda_5 = zeros(num_simulations, 1); % GMM (continuously updating)
    
    % Run simulations
    for sim = 1:num_simulations
        % Generate data from Poisson distribution
        X = poissrnd(lambda_true, n, 1);
        
        % Estimate lambda using each method
        lambda_1(sim) = mom_first_moment(X); % MOM (1st moment)
        lambda_2(sim) = mom_second_moment(X); % MOM (2nd moment)
        lambda_3(sim) = gmm_uniform_weight(X); % GMM (uniform)
        lambda_4(sim) = gmm_two_step(X); % GMM (two-step)
        lambda_5(sim) = gmm_continuously_updating(X); % GMM (continuously updating)
    end
    
    % Compute statistics for each estimator
    lambda_hat = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5]; % Combine estimates
    lambda_mean = mean(lambda_hat); % Mean of estimates
    lambda_bias = lambda_mean - lambda_true; % Bias of estimates
    lambda_variance = var(lambda_hat); % Variance of estimates
    lambda_std = sqrt(lambda_variance); % Standard deviation of estimates
    lambda_se = lambda_std / sqrt(num_simulations); % Standard error of estimates
    
    
 Display results in a table
    estimator_names = {'MOM (1st moment)', 'MOM (2nd moment)', 'GMM (uniform)', 'GMM (two-step)', 'GMM (continuously updating)'};
    disp(['<strong> Sample size n = ', num2str(n), ' </strong>']);
    disp(table(lambda_mean', lambda_bias', lambda_variance', lambda_std', lambda_se', ...
        'VariableNames', {'MeanEstimate', 'Bias', 'Variance', 'StandardDeviation', 'StandardError'}, ...
...
'RowNames', estimator_names));
    
     
Visualization of the distribution of estimates
    figure;
    sgtitle(['Distribution of Estimates for Sample Size n = ', num2str(n)]);
    
    % Histogram for MOM (1st moment)
    subplot(2, 3, 1);
    histogram(lambda_1, 'Normalization', 'pdf', 'FaceColor', 'b');
    xline(lambda_true, 'r', 'LineWidth', 2); % Add vertical line at true value
    title('MOM (1st moment)');
    xlabel('$$\hat{\lambda}$$', 'Interpreter', 'latex');
    ylabel('Density');
    grid on;
    
    % Histogram for MOM (2nd moment)
    subplot(2, 3, 2);
    histogram(lambda_2, 'Normalization', 'pdf', 'FaceColor', 'r');
    xline(lambda_true, 'r', 'LineWidth', 2); % Add vertical line at true value
    title('MOM (2nd moment)');
    xlabel('$$\hat{\lambda}$$', 'Interpreter', 'latex');
    ylabel('Density');
    grid on;
    
    % Histogram for GMM (uniform)
    subplot(2, 3, 3);
    histogram(lambda_3, 'Normalization', 'pdf', 'FaceColor', 'g');
    xline(lambda_true, 'r', 'LineWidth', 2); % Add vertical line at true value
    title('GMM (uniform)');
    xlabel('$$\hat{\lambda}$$', 'Interpreter', 'latex');
    ylabel('Density');
    grid on;
    
    % Histogram for GMM (two-step)
    subplot(2, 3, 4);
    histogram(lambda_4, 'Normalization', 'pdf', 'FaceColor', 'm');
    xline(lambda_true, 'r', 'LineWidth', 2); % Add vertical line at true value
    title('GMM (two-step)');
    xlabel('$$\hat{\lambda}$$', 'Interpreter', 'latex');
    ylabel('Density');
    grid on;
    
    % Histogram for GMM (continuously updating)
    subplot(2, 3, 5);
    histogram(lambda_5, 'Normalization', 'pdf', 'FaceColor', 'c');
    xline(lambda_true, 'r', 'LineWidth', 2); % Add vertical line at true value
    title('GMM (continuously updating)');
    xlabel('$$\hat{\lambda}$$', 'Interpreter', 'latex');
    ylabel('Density');
    grid on;
end

