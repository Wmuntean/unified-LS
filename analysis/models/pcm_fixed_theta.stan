// Stan Model for the Partial Credit Model (PCM)
// Assumes person abilities (theta) are known and passed in as data.
// This model estimates the item step difficulty parameters.
// It handles items with varying numbers of score categories.

data {
  // --- Dimensions ---
  int<lower=1> I;               // Number of items
  int<lower=1> P;               // Number of persons
  int<lower=1> N;               // Number of total observations (person-item pairs)
  int<lower=2> K_max;           // Maximum number of score categories for any item

  // --- Data Indices ---
  array[N] int<lower=1, upper=P> pp;  // Person index for each observation
  array[N] int<lower=1, upper=I> ii;  // Person index for each observation

  // --- Observed Data ---
  // Responses must be coded as integers 1, 2, ..., K[i]
  array[N] int<lower=1, upper=K_max> resp;

  // Number of score categories for each item (e.g., 2 for dichotomous)
  array[I] int<lower=1, upper=K_max> K;

  // Known person parameters
  vector[P] theta;
}

parameters {
  // Item step difficulty parameters.
  // We estimate K_max-1 for each item, but only use K[i]-1 for item i.
  matrix[I, K_max - 1] delta;
}

model {
  // --- Priors ---
  // A weakly informative prior on all step difficulty parameters.
  // to_vector() flattens the matrix into a single vector for efficient prior assignment.
  to_vector(delta) ~ normal(0, 5);

  // --- Likelihood ---
  // Loop through each observation to compute the PCM probability
  for (n in 1:N) {
    int item_idx = ii[n];         // Get the item for this observation
    int person_idx = pp[n];       // Get the person for this observation
    int num_cats = K[item_idx];   // Get the number of categories for this specific item

    vector[num_cats] eta;         // Vector to hold the logits

    // This block directly implements the cumulative logic of the PCM
    // It is the Stan equivalent of your Python function's core logic.
    eta[1] = 0; // The logit for the first category is the reference, set to 0.
    for (k in 2:num_cats) {
      // The logit for category k is the logit for k-1 plus (theta - step_k-1)
      eta[k] = eta[k - 1] + theta[person_idx] - delta[item_idx, k - 1];
    }

    // The categorical_logit distribution takes the vector of logits (eta)
    // and handles the normalization (softmax) internally in a stable way.
    // This is the most efficient method.
    resp[n] ~ categorical_logit(eta);
  }
}