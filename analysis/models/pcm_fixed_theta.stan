data {
  int<lower=1> N; // Total observations
  array[N] int<lower=1> item_id;
  array[N] int<lower=1> person_id;
  array[N] real theta;
  array[N] int<lower=1> scores;
  array[N] int<lower=2> categories_per_item;
  
  int<lower=1> n_items; // Number of unique items
  array[n_items] int<lower=1> threshold_start; // Starting index for each item's thresholds
  int<lower=1> total_thresholds; // Total number of threshold parameters
  int<lower=2> max_categories; // max score categories across all items
}
parameters {
  vector[total_thresholds] threshold; // Single vector containing all thresholds
}
model {
  // Prior on all thresholds
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
