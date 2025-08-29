// -----------------------------------------------------------------------------
// Model: Latent-Space Zero-Inflated Negative Binomial (LS-ZINB) Model
// Author: William Muntean
// Description:
//   Stan implementation of a Latent-Space Zero-Inflated Negative Binomial model.
//   - Estimates person parameters for both the count (theta) and zero-inflation (omega) processes.
//   - Estimates item parameters for count (b) and zero-inflation (eta).
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
//   - theta: Person ability for the count process
//   - omega: Person propensity for the non-zero process
//   - b: Item difficulty for the count process
//   - eta: Item propensity for the zero-inflation process
//   - zt: Item latent positions
//   - xi: Person latent positions
//   - log_gamma: Latent space distance multiplier
//   - phi: Negative binomial dispersion parameter
// -----------------------------------------------------------------------------

data {
  // --- Dimensions & Indices ---
  int<lower=1> N; // Total observations
  int<lower=1> n_items; // Number of unique items
  int<lower=1> n_persons; // Number of unique persons
  int<lower=1> D; // Number of latent dimensions
  
  // --- Data & Indices ---
  array[N] int<lower=1> item_id;
  array[N] int<lower=1> person_id;
  array[N] int<lower=0> process_counts; // The observed count data (e.g., answer changes)
}
parameters {
  // --- Person Parameters ---
  vector[n_persons] theta; // Person ability for the count process
  vector[n_persons] omega; // Person propensity for the non-zero process
  
  // --- Item Parameters ---
  vector[n_items] b; // Item difficulty for the count process
  vector[n_items] eta; // Item propensity for the zero-inflation process
  
  // --- Latent Space Parameters ---
  matrix[n_items, D] zt; // Raw item positions
  matrix[n_persons, D] xi; // Person positions
  real log_gamma; // Log of the distance multiplier
  
  // --- Negative Binomial Dispersion ---
  real<lower=0> phi; // Dispersion parameter
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
  // Person parameters are fixed to standard normal for identifiability
  theta ~ std_normal();
  omega ~ std_normal();
  
  // Weakly informative priors for item parameters
  b ~ normal(0, 2);
  eta ~ normal(0, 2);
  
  // Priors for latent space parameters
  to_vector(zt) ~ std_normal();
  to_vector(xi) ~ std_normal();
  log_gamma ~ normal(0, 1);
  
  // Prior for the dispersion parameter
  phi ~ cauchy(0, 2); // A weakly informative prior for a positive parameter
  
  // --- Likelihood ---
  real gamma = exp(log_gamma);
  
  for (n in 1 : N) {
    int item = item_id[n];
    int person = person_id[n];
    int y = process_counts[n];
    
    // --- ZINB Mixture Model Logic ---
    // 1. Calculate the logit of the probability of being in the count model (pi)
    real logit_pi = omega[person] - eta[item];
    
    // 2. Calculate the log of the expected count (mu) for the Negative Binomial
    real dist = distance(xi[person], zt_centered[item]);
    real log_mu = (theta[person] - gamma * dist) - b[item];
    
    // 3. Manually implement the log-likelihood for a zero-inflated model
    if (y == 0) {
      // If the count is 0, it could be a structural zero OR a count of 0.
      // log P(y=0) = log( P(structural zero) + P(count=0) )
      target += log_sum_exp(bernoulli_logit_lpmf(0 | logit_pi),
                            // Log prob of structural zero
                            bernoulli_logit_lpmf(1 | logit_pi)
                            + // Log prob of being in count model...
                            neg_binomial_2_log_lpmf(0 | log_mu, phi)); // ...times log prob of count being 0
    } else {
      // If the count is > 0, it must have come from the count model.
      // log P(y>0) = log( P(in count model) * P(count=y) )
      target += bernoulli_logit_lpmf(1 | logit_pi)
                + // Log prob of being in count model...
                neg_binomial_2_log_lpmf(y | log_mu, phi); // ...plus log prob of the specific count
    }
  }
}
