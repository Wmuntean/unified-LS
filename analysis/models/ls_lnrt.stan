// -----------------------------------------------------------------------------
// Model: Latent-Space Log-Normal Response Time (LS-LNRT) Model
// Author: William Muntean
// Description:
//   Stan implementation of a Latent-Space Log-Normal Response Time model.
//   - Estimates person speededness (tau) and item time parameters (beta, alpha).
//   - Latent space structure penalizes response time by person-item distance.
//   - Suitable for use with CmdStanPy and pandas.
//
// Inputs:
//   --- Dimensions & Indices ---
//   - N: Total observations
//   - n_items: Number of unique items
//   - n_persons: Number of unique persons
//   - D: Number of latent dimensions
//
//   --- Data & Indices ---
//   - array[N] int<lower=1> item_id: Item index for each response (1-based)
//   - array[N] int<lower=1> person_id: Person index for each response (1-based)
//   - vector[N] log_rt: Pre-log-transformed response times
//
// Output:
//   - tau: Person speededness parameter
//   - beta: Item time intensity (slowness)
//   - log_alpha: Log of item time discrimination
//   - zt: Item latent positions
//   - xi: Person latent positions
//   - log_gamma: Latent space distance multiplier
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
  vector[N] log_rt;
}
parameters {
  // --- Item Parameters ---
  vector[n_items] beta;
  vector[n_items] log_alpha;
  matrix[n_items, D] zt;
  
  // --- Person Parameters ---
  vector[n_persons] tau;
  matrix[n_persons, D] xi;
  
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
  tau ~ std_normal();
  beta ~ normal(0, 2);
  log_alpha ~ normal(0, 1);
  vector[n_items] alpha = exp(log_alpha);

  to_vector(zt) ~ std_normal();
  to_vector(xi) ~ std_normal();
  
  // Prior from https://doi.org/10.1007/s11336-021-09762-5
  // Prior from https://doi.org/10.3390/jintelligence12040038
  log_gamma ~ normal(0.5, 1);
  real gamma = exp(log_gamma);
  
  // --- Likelihood ---
  for (n in 1 : N) {
    int item = item_id[n];
    int person = person_id[n];
    
    // Calculate the distance in the latent space
    real dist = distance(xi[person], zt_centered[item]);
    
    // Define the mean and standard deviation for the log-normal model
    real mu = beta[item] - tau[person] + gamma * dist;
    real sigma = 1 / alpha[item];
    
    // The likelihood statement
    log_rt[n] ~ normal(mu, sigma);
  }
}
