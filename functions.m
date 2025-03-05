% -------------------------------------------------------------------------
% Functions for Estimating Lambda
% -------------------------------------------------------------------------

% 1. MOM Estimator Using the First Moment Condition
function lambda_hat = mom_first_moment(X)
    lambda_hat = mean(X);
end

% 2. MOM Estimator Using the Second Moment Condition
function lambda_hat = mom_second_moment(X)
    mean_X_squared = mean(X.^2);
    a = 1;
    b = 1;
    c = -mean_X_squared;
    lambda_hat = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);
end

% 3. GMM Estimator Using a Uniform (Identity) Weighting Matrix
function lambda_hat = gmm_uniform_weight(X)
    lambda_initial = mean(X);
    options = optimoptions('fminunc', 'Display', 'off');
    lambda_hat = fminunc(@(lambda) gmm_objective_uniform(lambda, X), lambda_initial, options);
end

function Q = gmm_objective_uniform(lambda, X)
    g1 = mean(X - lambda);
    g2 = mean(X.^2 - lambda^2 - lambda);
    g = [g1; g2];
    W = eye(2); % Identity weighting matrix
    Q = g' * W * g;
end

% 4. GMM Estimator Using a Two-Step Feasible Weighting Matrix
function lambda_hat = gmm_two_step(X)
    n = length(X);
    lambda_initial = mean(X);
    g1 = X - lambda_initial;
    g2 = X.^2 - lambda_initial^2 - lambda_initial;
    g = [g1, g2];
    W = inv((g' * g) / n);
    options = optimoptions('fminunc', 'Display', 'off');
    lambda_hat = fminunc(@(lambda) gmm_objective_two_step(lambda, X, W), lambda_initial, options);
end

function Q = gmm_objective_two_step(lambda, X, W)
    g1 = mean(X - lambda);
    g2 = mean(X.^2 - lambda^2 - lambda);
    g = [g1; g2];
    Q = g' * W * g;
end

% 5. GMM Estimator Using a Continuously Updating Weighting Matrix
function lambda_hat = gmm_continuously_updating(X)
    n = length(X);
    lambda_initial = mean(X);
    options = optimoptions('fminunc', 'Display', 'off');
    lambda_hat = fminunc(@(lambda) gmm_objective_continuously_updating(lambda, X), lambda_initial, options);
end

function Q = gmm_objective_continuously_updating(lambda, X)
    n = length(X);
    g1 = X - lambda;
    g2 = X.^2 - lambda^2 - lambda;
    g = [g1, g2];
    W = inv((g' * g) / n + 1e-6 * eye(2)); % Regularized weighting matrix
    g_bar = [mean(g1); mean(g2)];
    Q = g_bar' * W * g_bar;
end