data {
  int N; 
  int N_sample; 
  int N_holder;
  int N_location;
  int N_type;
  int N_track_id;

  int N_size;

  array[N] int type_ids;
  array[N] int sample_ids;
  array[N] int holder_ids;
  array[N] int location_ids;

  array[N] int size_ids;

  array[N] int track_id_ids;
  array[N] real y;
}
parameters {

  vector[N_type] mu;
  vector<lower=0>[N_type] sigma;
  vector<lower=0>[N_type] sigma_sample; 
  vector<lower=0>[N_type] sigma_holders;
  
  vector<lower=0>[N_size] sigma_size;

  vector[N_type] z_types; 
  vector[N_holder] z_holders; 
  vector[N_sample] z_samples;

  vector[N] z_size;

  // real<lower=0> sigma_mu;
  // real<lower=0> sigma_sigma;

  vector<lower=0>[N_track_id] sigma_common;
}
transformed parameters {
  vector[N] mu_type = mu[type_ids] + sigma[type_ids].*z_types[type_ids];
  vector[N] mu_sample = sigma_sample[type_ids].*z_samples[sample_ids];
  vector[N] mu_holder = sigma_holders[type_ids].*z_holders[holder_ids];

  vector[N] mu_size = sigma_size[size_ids].*z_size[track_id_ids];
}
model{

  mu ~ std_normal();
  sigma ~ std_normal();
  sigma_sample ~ std_normal();
  sigma_holders ~ std_normal();

  sigma_size ~ std_normal();
  
  z_types ~ std_normal();
  z_samples ~ std_normal();
  z_holders ~ std_normal();

  z_size ~ std_normal();

  // sigma_mu ~  normal(1,0.1);
  // sigma_sigma ~ normal(2,0.1);
  sigma_common ~ inv_gamma(1,2);

  y ~ normal(mu_type+mu_sample+mu_holder + mu_size, sigma_common[track_id_ids]);
}
generated quantities {
 vector[N] log_likelihood;
 array[N] real y_hat;

 for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu_type[i]+mu_sample[i]+mu_holder[i]+mu_size[i],sigma_common[track_id_ids[i]]);
 } 
 y_hat = normal_rng(mu_type+mu_sample+mu_holder+mu_size,sigma_common[track_id_ids]);
}