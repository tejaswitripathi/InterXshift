library(InterXshift)
library(devtools)
library(kableExtra)
library(sl3)
library(nnls)
library(dplyr)

seed <- 429153
set.seed(seed)


data("NIEHS_data_1", package = "InterXshift")
NIEHS_data_1$W <- rnorm(nrow(NIEHS_data_1), mean = 0, sd = 0.1)
cols <- c("X2", "X3", "X4", "X6", "X7")
NIEHS_data_1 <- NIEHS_data_1 %>% select(-all_of(cols))
w <- NIEHS_data_1[, c("W", "Z")]
a <- NIEHS_data_1[, c("X1", "X5")]
y <- NIEHS_data_1$Y
data_internal <- data.table::data.table(w, a, y)
head(NIEHS_data_1)
head(w)
head(a)
head(y)
head(data_internal)
a <- data.frame(a)
w <- data.frame(w)

n_folds = 3
outcome_type = "continuous"
sls <- create_sls()
mu_learner <- sls$mu_learner

data_internal$folds <- create_cv_folds(n_folds, data_internal$y)

psi <- list()

delta_values_x1 <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

for (i in delta_values_x1){

  deltas <- list(
    "X1" = i,
    "X5" = 1
  )



  fold_basis_results <- furrr::future_map(unique(data_internal$folds),
                                          function(fold_k) {
                                            at <- data_internal[data_internal$folds != fold_k, ]
                                            av <- data_internal[data_internal$folds == fold_k, ]

                                            intxn_results <- find_synergy_antagonism(
                                              data = at,
                                              deltas = deltas,
                                              a_names = colnames(a),
                                              w_names = colnames(w),
                                              outcome = "y",
                                              outcome_type = outcome_type,
                                              mu_learner = mu_learner,
                                              seed = seed,
                                              top_n = 1
                                            )

                                          },
                                          .options = furrr::furrr_options(seed = seed, packages = "InterXshift")
  )

  fold_intxn_results <- fold_basis_results[[1]]
  fold_positive_effects <- fold_intxn_results$top_positive_effects
  at <- data_internal[data_internal$folds != 1, ]
  av <- data_internal[data_internal$folds == 1, ]

  exposure <- fold_positive_effects$Variable[1]

  lower_bound <- min(min(av[[exposure]]), min(at[[exposure]]))
  upper_bound <- max(max(av[[exposure]]), max(at[[exposure]]))

  delta <- deltas[[exposure]]
  pi_learner <- sls$pi_learner

  ind_gn_exp_estim <- estimate_density_ratio(at = at, av = av, delta =  delta, var = exposure, covars = c(exposure, colnames(w)), classifier = mu_learner)

  covars <- c(colnames(a), colnames(w))

  ind_qn_estim <- try(indiv_stoch_shift_est_Q(
    exposure = exposure,
    delta = delta,
    mu_learner = mu_learner,
    covars = covars,
    av = av,
    at = at,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    outcome_type = "continuous"
  ))
  if (inherits(ind_qn_estim, "try-error")) {
    # Print a message indicating an error occurred
    print(paste("Error on iteration", i))
  }

  Hn <- ind_gn_exp_estim$Hn_av

  tmle_fit <- tmle_exposhift(
    data_internal = av,
    delta = delta,
    Qn_scaled = ind_qn_estim$q_av,
    Qn_unscaled = scale_to_original(ind_qn_estim$q_av, min_orig = min(av$y), max_orig = max(av$y)),
    Hn = Hn,
    fluctuation = "standard",
    y = av$y
  )

  pos_rank_shift_in_fold <- calc_final_ind_shift_param(
    tmle_fit,
    exposure,
    1
  )

  pos_rank_fold_results <- list()
  pos_rank_fold_results[[
    paste("Rank", 1, ":", exposure)
  ]] <- list(
    "data" = av,
    "Qn_scaled" = ind_qn_estim$q_av,
    "Hn" = Hn,
    "k_fold_result" = pos_rank_shift_in_fold,
    "Delta" = delta
  )

  psi <- append(psi, pos_rank_fold_results$`Rank 1 : X1`$k_fold_result$Psi)

}

print(psi)

# Define the MSM function
msm <- function(delta, b0, b1) {
  return(b0 + b1 * delta)
}

# Define the loss function
loss <- function(params, psi, delta_values_x1) {
  b0 <- params[1]
  b1 <- params[2]
  sum = 0
  for (i in 1:length(psi)) {
    sum = sum + ((psi[[i]] - msm(delta_values_x1[[i]], b0, b1)))**2
  }
  return(sum)
}

# Minimize the loss function
opt_result <- optim(par = c(0, 0), fn = loss, psi = psi, delta_values_x1 = delta_values_x1)

# Print the optimized parameters
print(opt_result$par)

# Print the minimized loss
print(opt_result$value)

