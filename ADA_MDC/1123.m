% Part 1: Ordinary Least Squares (unchanged)

% Load the data from the text file into an array
data = load('3060309_MDC1.txt');    % read two‑column data (x,y)
x = data(:,1);                      % extract x values
y = data(:,2);                      % extract y values
n = length(x);                      % count data points

% Compute the sample means of x and y
x_bar = mean(x);                    % mean of x
y_bar = mean(y);                    % mean of y

% Compute sums of squared and cross deviations
S_xx = sum((x - x_bar).^2);         % Σ(x - x̄)²
S_xy = sum((x - x_bar) .* (y - y_bar)); % Σ(x - x̄)(y - ȳ)

% Calculate the OLS slope and intercept estimators
b_hat = S_xy / S_xx;                % slope = S_xy / S_xx
a_hat = y_bar - b_hat * x_bar;      % intercept = ȳ - b̂ x̄

% Compute residuals and estimate noise variance
res = y - (a_hat + b_hat * x);      % residuals y - (a + b x)
SSR = sum(res.^2);                  % Σ residual²
sigma2_hat = SSR / (n - 2);         % SSR/(n-2)

% Variance and covariance of estimators
var_b = sigma2_hat / S_xx;                          % σ²/S_xx
var_a = sigma2_hat * (1/n + x_bar^2 / S_xx);        % σ²(1/n + x̄²/S_xx)
cov_ab = -x_bar * sigma2_hat / S_xx;                % -x̄ σ²/S_xx
se_a = sqrt(var_a);                                 % √var_a
se_b = sqrt(var_b);                                 % √var_b

% Print the key OLS results
fprintf('Number of data points: %d\n', n);
fprintf('Estimated slope (b): %.5f\n', b_hat);
fprintf('Estimated intercept (a): %.5f\n', a_hat);
fprintf('Estimated noise variance (σ²): %.5f\n', sigma2_hat);
fprintf('Standard error of a: %.5f\n', se_a);
fprintf('Standard error of b: %.5f\n', se_b);
fprintf('Covariance of (a, b): %.5f\n', cov_ab);

% Plot data and fitted line
figure('Position',[100 100 800 600]);
scatter(x, y, 60, 'MarkerFaceColor',[0.5 0 0], 'MarkerEdgeColor','none','MarkerFaceAlpha',0.7);
hold on;
x_line = [min(x), max(x)];                             % endpoints for line
y_line = a_hat + b_hat * x_line;                       % y = a + b x
plot(x_line, y_line, '--', 'LineWidth',2, 'Color',[1 0.647 0]);
hold off;
xlabel('x','FontSize',14);
ylabel('y','FontSize',14);
title('Ordinary Least Squares Fit','FontSize',16);
legend('Data Points', sprintf('Fit: y = %.3f + %.3fx', a_hat, b_hat),'FontSize',12);
grid on;
grid minor;


%% Part 2: Grid search and χ² contours

% helper for constant errors
sigma = ones_like(y);              % constant errors array
N = length(x);                     % data count

% 2. Manual least‑squares estimates for a0 (slope) and b0 (intercept)
x_mean = mean(x);                  
y_mean = mean(y);
Sxx = sum((x - x_mean).^2);
Sxy = sum((x - x_mean) .* (y - y_mean));
a0 = Sxy / Sxx;                    % slope estimate
b0 = y_mean - a0 * x_mean;         % intercept estimate

% residual variance & parameter uncertainties
resid = y - (a0 * x + b0);
chi2_ls = sum((resid ./ sigma).^2);
sigma2_resid = chi2_ls / (N - 2);
sigma_a = sqrt(sigma2_resid / Sxx);
sigma_b = sqrt(sigma2_resid * (1/N + x_mean^2 / Sxx));

% 3. Build (a,b) grid ±3σ around LS solution
n_a = 200; n_b = 200;
a_vals = linspace(a0 - 3*sigma_a, a0 + 3*sigma_a, n_a);
b_vals = linspace(b0 - 3*sigma_b, b0 + 3*sigma_b, n_b);
[A, B] = meshgrid(a_vals, b_vals);

% 4. Compute χ² on the grid
Chi2 = zeros(size(A));
for i = 1:n_b
    for j = 1:n_a
        res_grid = y - (A(i,j) * x + B(i,j));
        Chi2(i,j) = sum((res_grid ./ sigma).^2);
    end
end

% 5. Find best‑fit and Δχ² using custom unravel_index
[chi2_min, flat_idx] = min(Chi2(:));
idx = unravel_index(flat_idx, size(Chi2));   % [row, col]
i_min = idx(1); j_min = idx(2);
a_best = A(i_min, j_min);
b_best = B(i_min, j_min);
delta_chi2 = Chi2 - chi2_min;

% 6. Compute credible‑region bounds
levels = [2.30, 6.17, 11.8];
region_ranges = zeros(numel(levels),4);
for k = 1:numel(levels)
    mask = delta_chi2 <= levels(k);
    region_ranges(k,:) = [min(A(mask)), max(A(mask)), min(B(mask)), max(B(mask))];
end

% 7. Print results
fprintf('\nBest‑fit parameters:\n');
fprintf('  a = %.5f\n', a_best);
fprintf('  b = %.5f\n', b_best);
fprintf('  χ²_min = %.3f\n\n', chi2_min);
perc = [68.3, 95.4, 99.73];
for k = 1:numel(levels)
    fprintf('%.1f%% credible region:\n', perc(k));
    fprintf('  a ∈ [%.5f, %.5f]\n', region_ranges(k,1), region_ranges(k,2));
    fprintf('  b ∈ [%.5f, %.5f]\n\n', region_ranges(k,3), region_ranges(k,4));
end

% 8. Plot Δχ² contours
figure('Position',[200 200 800 600]);
[C,h] = contour(A, B, delta_chi2, levels, 'LineWidth',1.5);
hold on;
plot(a_best, b_best, 'k+', 'DisplayName','Best‑fit');
hold off;
xlabel('a');
ylabel('b');
title('\Delta\chi^2 Contours for (a,b)');
clabel(C,h,{'68.3%','95.4%','99.73%'});
legend;


%% Part 3: Metropolis MCMC using OLS parameters

% Log‑likelihood function using σ² from the OLS stage
log_likelihood = @(a,b) -0.5 * sum((y - (a + b*x)).^2) / sigma2_hat;

% Define uniform priors ±5 around a_hat and ±3 around b_hat
a_min = a_hat - 5.0;  a_max = a_hat + 5.0;
b_min = b_hat - 3.0;  b_max = b_hat + 3.0;

% Log‑prior and log‑posterior
log_prior = @(a,b) (a>=a_min && a<=a_max && b>=b_min && b<=b_max).*0 + ~(a>=a_min && a<=a_max && b>=b_min && b<=b_max).*-Inf;
log_posterior = @(a,b) log_prior(a,b) + log_likelihood(a,b);

% MCMC hyperparameters
prop_a = 0.05; prop_b = 0.02;
steps = 40000; burn = 10000; thin = 10;

% Initialize chain
current_a = a_hat; current_b = b_hat;
current_lp = log_posterior(current_a, current_b);
chain = zeros(steps,2);
accept = 0;
rng(0);

for t = 1:steps
    cand_a = current_a + prop_a * randn;
    cand_b = current_b + prop_b * randn;
    cand_lp = log_posterior(cand_a, cand_b);
    if rand < exp(cand_lp - current_lp)
        current_a = cand_a;
        current_b = cand_b;
        current_lp = cand_lp;
        accept = accept + 1;
    end
    chain(t,:) = [current_a, current_b];
end

posterior = chain(burn+1:thin:end, :);
mean_a = mean(posterior(:,1));  mean_b = mean(posterior(:,2));
std_a  = std(posterior(:,1), 1);  std_b  = std(posterior(:,2), 1);

% Print MCMC summary
fprintf('\nMCMC results:\n');
fprintf('Acceptance rate: %.4f\n', accept/steps);
fprintf('Posterior mean a = %.5f ± %.5f\n', mean_a, std_a);
fprintf('Posterior mean b = %.5f ± %.5f\n', mean_b, std_b);

% Corner‑style plots
figure;
histogram(posterior(:,1),40,'Normalization','pdf','EdgeColor','black','LineWidth',1.2);
hold on; xline(mean_a,'--k'); hold off;
title('Posterior of parameter a');
xlabel('a'); ylabel('Probability density');

figure;
histogram(posterior(:,2),40,'Normalization','pdf','EdgeColor','black','LineWidth',1.2);
hold on; xline(mean_b,'--k'); hold off;
title('Posterior of parameter b');
xlabel('b'); ylabel('Probability density');

% Joint posterior contour
[H, xedges, yedges] = histcounts2(posterior(:,1),posterior(:,2),50,'Normalization','pdf');
xcenters = (xedges(1:end-1)+xedges(2:end))/2;
ycenters = (yedges(1:end-1)+yedges(2:end))/2;
Hsort = sort(H(:),'descend');
Hcum = cumsum(Hsort)/sum(Hsort);
level68 = Hsort(find(Hcum>=0.683,1));
level95 = Hsort(find(Hcum>=0.954,1));

figure;
contourf(xcenters, ycenters, H', [level95, level68, max(H(:))]);
hold on;
scatter(posterior(:,1), posterior(:,2), 4, 'filled','MarkerFaceAlpha',0.4);
hold off;
xlabel('a'); ylabel('b');
title('Joint posterior of a and b');


%% Local helper functions

function out = ones_like(arr)
    out = ones(size(arr));
end

function idx = unravel_index(flat_index, shape)
    % Convert a linear index into subscripts for an array of size 'shape'
    rem = flat_index - 1;
    dims = numel(shape);
    idx = zeros(1,dims);
    for k = dims:-1:1
        idx(k) = mod(rem, shape(k));
        rem = floor(rem / shape(k));
    end
    idx = idx + 1;
end
