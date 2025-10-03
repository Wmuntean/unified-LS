// -----------------------------------------------------------------------------
// Model: Latent Space Partial Credit Model (LS-PCM) with Mixture Prior (PIP)
// Author: William Muntean
// Description:
//   Stan implementation of the Latent Space Partial Credit Model for polytomous IRT data.
//   - Estimates item thresholds and latent positions for persons and items.
//   - Person abilities (theta) are fixed and provided as data.
//   - Latent space structure penalizes effective theta by person-item distance.
//   - Mixture prior on latent space effect (log_gamma) with posterior inclusion probability (pip).
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
//   - theta: Fixed ability for each response
//   - scores: Observed score for each response (1-based)
//   - categories_per_item: Number of score categories for each item
//   - threshold_start: Starting index for each item's thresholds in the threshold vector
//   - total_thresholds: Total number of thresholds across all items
//   - max_categories: Maximum number of categories for any item
//
// Output:
//   - threshold: Estimated item thresholds
//   - zt: Item latent positions
//   - xi: Person latent positions
//   - log_gamma: Latent space distance multiplier
//   - pip: Posterior inclusion probability for latent space effect
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
}
parameters {
  // --- Item Parameters ---
  vector[total_thresholds] threshold;
  matrix[n_items, D] zt;
  
  // --- Person Parameters ---
  matrix[n_persons, D] xi;
  
  // --- Latent Space Parameters ---
  real log_gamma;
  real<lower=0, upper=1> pip; // Posterior inclusion probability
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
  to_vector(zt) ~ std_normal();
  to_vector(xi) ~ std_normal();
  pip ~ beta(1, 1);
  
  // Mixture components for log_gamma.
  // Prior from https://doi.org/10.3390/jintelligence12040038
  real log_prob_spike = log(1 - pip) + normal_lpdf(log_gamma | -5, 1);
  real log_prob_slab = log(pip) + normal_lpdf(log_gamma | .5, 1);
  target += log_sum_exp(log_prob_spike, log_prob_slab);
  real gamma = exp(log_gamma);
  
  vector[max_categories] eta;
  // --- Likelihood ---
  for (n in 1 : N) {
    int item = item_id[n];
    int person = person_id[n];
    int score = scores[n];
    int n_cats = categories_per_item[n];
    int idx_start = threshold_start[item];
    
    // --- Core Latent Space Calculation ---
    // 1. Calculate the Euclidean distance for this person-item pair
    //    using the *centered* item positions.
    real dist = distance(xi[person], zt_centered[item]);
    // real dist = distance(xi[person], zt[item]);
    
    // 2. Calculate the "effective theta" penalized by the distance
    real effective_theta = theta[n] - gamma * dist;
    
    // --- PCM Likelihood (using effective_theta) ---
    eta[1] = 0;
    for (k in 2 : n_cats) {
      eta[k] = eta[k - 1] + effective_theta - threshold[idx_start + k - 2];
    }
    
    target += eta[score] - log_sum_exp(head(eta, n_cats));
  }
}
