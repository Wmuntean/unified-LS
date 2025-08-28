// -----------------------------------------------------------------------------
// Model: Partial Credit Model (PCM) with Fixed Theta
// Author: William Muntean
// Description:
//   Stan implementation of the Partial Credit Model for polytomous IRT data.
//   Item thresholds are parameterized as a single vector. Person abilities (theta)
//   are fixed and provided as data. Suitable for use with CmdStanPy and pandas.
//
// Inputs:
//   --- Dimensions & Indices ---
//   - N: Number of observations
//   - n_items: Number of unique items
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
// -----------------------------------------------------------------------------
data {
  // --- Dimensions & Indices ---
  int<lower=1> N;
  int<lower=1> n_items;
  
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
}
model {
  // --- Priors ---
  threshold ~ normal(0, 2);
  
  vector[max_categories] eta;
  
  // Likelihood
  for (n in 1 : N) {
    int item = item_id[n];
    int score = scores[n];
    int n_cats = categories_per_item[n];
    int idx_start = threshold_start[item];
    
    eta[1] = 0;
    for (k in 2 : n_cats) {
      eta[k] = eta[k - 1] + theta[n] - threshold[idx_start + k - 2];
    }
    
    // score ~ categorical_logit(eta);
    target += eta[score] - log_sum_exp(head(eta, n_cats));
  }
}
