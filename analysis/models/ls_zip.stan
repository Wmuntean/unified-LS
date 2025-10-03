// -----------------------------------------------------------------------------
// Model: Latent-Space Zero-Inflated Poisson (LS-ZIP) Model
// Author: William Muntean
// Description:
//   Stan implementation of a Latent-Space Zero-Inflated Poisson model.
//   - Estimates person parameters for both the count (kappa) and zero-inflation (omega) processes.
//   - Estimates item parameters for count (delta) and zero-inflation (lambda).
//   - Latent space structure penalizes expected counts by person-item distance.
//   - Suitable for use with CmdStanPy and pandas.
//
// Inputs:
//   --- Dimensions & Indices ---
//   - N: Number of observations
//   - n_items: Number of unique items
//   - n_persons: Number of unique persons
//   - D: Number of latent dimensions
//
//   --- Data & Indices ---
//   - item_id: Item index for each response (1-based)
//   - person_id: Person index for each response (1-based)
//   - process_counts: Observed count data (e.g., answer changes)
//
// Output:
//   - kappa: Person ability for the count process
//   - omega: Person propensity for the non-zero process
//   - delta: Item difficulty for the count process
//   - lambda: Item propensity for the zero-inflation process
//   - zt: Item latent positions
//   - xi: Person latent positions
//   - log_gamma: Latent space distance multiplier
// -----------------------------------------------------------------------------
// Stan Model for a Latent-Space Zero-Inflated Poisson (LS-ZIP) Ideal Point Model
// This baseline version uses a single gamma parameter.

data {
  // --- Dimensions & Indices ---
  int<lower=1> N;
  int<lower=1> n_items;
  int<lower=1> n_persons;
  int<lower=1> D;
  
  // --- Data & Indices ---
  array[N] int<lower=1> item_id;
  array[N] int<lower=1> person_id;
  array[N] int<lower=0> process_counts;
}
parameters {
  // --- Person Parameters ---
  vector[n_persons] kappa;
  vector[n_persons] omega;
  matrix[n_persons, D] xi;
  
  // --- Item Parameters ---
  vector[n_items] delta;
  vector[n_items] item_intercept;
  vector[n_items] lambda;
  matrix[n_items, D] zt;
  
  // --- Latent Space Parameters ---
  real log_gamma;
}
transformed parameters {
  // --- Centered Item Positions for Identifiability ---
  matrix[n_items, D] zt_centered;
  for (d in 1 : D) {
    zt_centered[ : , d] = zt[ : , d] - mean(zt[ : , d]);
  }
}
model {
  // --- Priors ---
  kappa ~ std_normal();
  omega ~ std_normal();
  
  // Weakly informative priors for item parameters
  delta ~ normal(0, 1.5);
  item_intercept ~ normal(1, 1);
  lambda ~ normal(0, 2);
  
  // Priors for latent space parameters
  to_vector(zt) ~ std_normal();
  to_vector(xi) ~ std_normal();
  
  // Prior from https://doi.org/10.1007/s11336-021-09762-5
  // Prior from https://doi.org/10.3390/jintelligence12040038
  log_gamma ~ normal(0.5, 1);
  
  // --- Likelihood ---
  real gamma = exp(log_gamma);
  
  for (n in 1 : N) {
    int item = item_id[n];
    int person = person_id[n];
    int y = process_counts[n];
    
    // --- ZIP Mixture ---
  real logit_pi = omega[person] - lambda[item];
    
    // Ideal Point Poisson
    real dist = distance(xi[person], zt_centered[item]);
  real quadratic_dist = pow(kappa[person] - delta[item], 2);
    
    // The expected log-count is the item's max, penalized by both the 
    // ideal point distance and the security latent space distance.
    real log_mu = item_intercept[item] - quadratic_dist - gamma * dist;
    
    if (y == 0) {
      target += log_sum_exp(bernoulli_logit_lpmf(0 | logit_pi),
                            bernoulli_logit_lpmf(1 | logit_pi)
                            + poisson_log_lpmf(0 | log_mu));
    } else {
      target += bernoulli_logit_lpmf(1 | logit_pi)
                + poisson_log_lpmf(y | log_mu);
    }
  }
}
