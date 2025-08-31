// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Model: Latent Space Unified Model (LS-Unified)
// Author: William Muntean
// Description:
//   Stan implementation of the LS-Unified model, which integrates three psychometric models:
//     - Partial Credit Model (PCM) for polytomous item response data
//     - Log-Normal Response Time (LNRT) model for response time data
//     - Zero-Inflated Negative Binomial (ZINB) model for count data (e.g., answer changes)
//   All three components share a common latent space structure and a single gamma parameter that modulates the effect of person-item distance across all submodels.
//   Person abilities (theta) are fixed and provided as data.
//   Suitable for use with CmdStanPy and pandas.
//
// Inputs:
//   --- Dimensions & Indices ---
//   - N: Number of observations
//   - n_items: Number of unique items
//   - n_persons: Number of unique persons
//   - D: Number of latent dimensions for the latent space
//
//   --- Data & Indices ---
//   - item_id: Item index for each response (1-based)
//   - person_id: Person index for each response (1-based)
//   - theta: Fixed ability for each response
//   - scores: Observed score for each response (1-based)
//   - categories_per_item: Number of score categories for each item
//   - threshold_start: Starting index for each item's thresholds in the threshold vector
//   - total_thresholds: Total number of thresholds across all items
//   - max_categories: Maximum number of categories for any item
//   - log_rt: Log-transformed response time for each response
//   - process_counts: Observed count data for each response (e.g., answer changes)
//
// Output Parameters:
//   - threshold: Estimated item thresholds (PCM)
//   - delta: Item parameter for ZINB (difficulty)
//   - lambda: Item parameter for ZINB (zero-inflation)
//   - beta: Item parameter for LNRT (location)
//   - log_alpha: Item parameter for LNRT (log-precision)
//   - zt: Item latent positions
//   - xi: Person latent positions
//   - kappa, omega, tau: Person parameters for ZINB and LNRT
//   - log_gamma: Latent space distance multiplier (shared across all submodels)
//   - phi: Negative binomial dispersion parameter (ZINB)
//
// Model Structure:
//   - Latent space coordinates for persons (xi) and items (zt) are estimated.
//   - The effect of person-item distance in the latent space is penalized by gamma and incorporated into all three submodels.
//   - PCM: Models polytomous item scores with thresholds and effective theta penalized by distance.
//   - LNRT: Models log-response times as a function of item and person parameters and latent space distance.
//   - ZINB: Models count data with zero-inflation and negative binomial likelihood, both modulated by latent space distance.
// -----------------------------------------------------------------------------
data {
  // --- Dimensions & Indices ---
  int<lower=1> N;
  int<lower=1> n_items;
  int<lower=1> n_persons;
  int<lower=1> D;
  
  // --- Data & Indices ---
  array[N] int<lower=1> item_id;
  array[N] int<lower=1> person_id;
  array[N] real theta;
  array[N] int<lower=1> scores;
  array[N] int<lower=2> categories_per_item;
  array[n_items] int<lower=1> threshold_start;
  int<lower=1> total_thresholds;
  int<lower=2> max_categories;
  vector[N] log_rt;
  array[N] int<lower=0> process_counts;
}
parameters {
  // --- Item Parameters ---
  vector[total_thresholds] threshold;
  vector[n_items] beta;
  vector[n_items] log_alpha;
  vector[n_items] delta;
  vector[n_items] lambda;
  matrix[n_items, D] zt;
  
  // --- Person Parameters ---
  vector[n_persons] kappa;
  vector[n_persons] omega;
  vector[n_persons] tau;
  matrix[n_persons, D] xi;
  
  // --- Latent Space Parameters ---
  real log_gamma;
  
  // --- Negative Binomial Dispersion ---
  real<lower=0> phi;
}
transformed parameters {
  // --- Centered Item Positions for Identifiability ---
  // This block centers the item latent space on the origin to prevent
  // the entire coordinate system from drifting during sampling (translation invariance).
  matrix[n_items, D] zt_centered;
  for (d in 1 : D) {
    zt_centered[ : , d] = zt[ : , d] - mean(zt[ : , d]);
  }
}
model {
  // --- Priors ---
  threshold ~ normal(0, 2);
  beta ~ normal(0, 2);
  log_alpha ~ normal(0, 1);
  vector[n_items] alpha = exp(log_alpha);
  delta ~ normal(0, 2);
  lambda ~ normal(0, 2);
  kappa ~ std_normal();
  omega ~ std_normal();
  tau ~ std_normal();
  
  to_vector(zt) ~ std_normal();
  to_vector(xi) ~ std_normal();
  
  // Prior from https://doi.org/10.1007/s11336-021-09762-5
  // Prior from https://doi.org/10.3390/jintelligence12040038
  log_gamma ~ normal(0.5, 1);
  real gamma = exp(log_gamma);
  
  phi ~ cauchy(0, 2);
  
  vector[max_categories] eta;
  // --- Likelihood ---
  for (n in 1 : N) {
    int item = item_id[n];
    int person = person_id[n];
    int score = scores[n];
    int n_cats = categories_per_item[n];
    int idx_start = threshold_start[item];
    int y = process_counts[n];
    
    // Calculate the distance in the latent space
    real dist = distance(xi[person], zt_centered[item]);
    
    // --- PCM Likelihood ---
    real effective_theta = theta[n] - gamma * dist;
    eta[1] = 0;
    for (k in 2 : n_cats) {
      eta[k] = eta[k - 1] + effective_theta - threshold[idx_start + k - 2];
    }
    target += eta[score] - log_sum_exp(head(eta, n_cats));
    
    // --- LNRT Likelihood ---
    real mu = beta[item] - tau[person] + gamma * dist;
    real sigma = 1 / alpha[item];
    log_rt[n] ~ normal(mu, sigma);
    
    // --- ZINB Likelihood ---
    real logit_pi = omega[person] - lambda[item];
    real log_mu = (kappa[person] - gamma * dist) - delta[item];
    
    if (y == 0) {
      target += log_sum_exp(bernoulli_logit_lpmf(0 | logit_pi),
                            bernoulli_logit_lpmf(1 | logit_pi)
                            + neg_binomial_2_log_lpmf(0 | log_mu, phi));
    } else {
      target += bernoulli_logit_lpmf(1 | logit_pi)
                + neg_binomial_2_log_lpmf(y | log_mu, phi);
    }
  }
}
