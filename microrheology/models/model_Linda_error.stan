data {
  int N; 
  int N_train;
  int N_samples; 
  int N_holders;
  int N_locations;
  int N_materials;
  int N_ids;

  array[N_samples] int material_ids;
  array[N_holders] int sample_ids;
  array[N_locations] int holder_ids;
  array[N_ids] int location_ids;

  array[N] int track_ids;
  array[N] int material_sigma_ids;

  array[N] real y;   
  array[N_train] int train_ids;
}
parameters {

  vector[N_materials] mu;
  vector<lower=0>[N_materials] sigma;
  vector<lower=0>[N_materials] sigma_sample; 
  vector<lower=0>[N_materials] sigma_holders;

  vector[N_holders] z_holders; 
  vector[N_samples] z_samples; 
  vector[N_ids] z_measurements; 

  // real mu_sigma;
  // real<lower=0> sigma_materials;
  // real<lower=0> sigma_common;

  real<lower=0> mu_sigma;
  real<lower=0> sigma_materials;

  vector<lower=0>[N_ids] sigma_common;
}
transformed parameters {
  vector[N_samples] mu_samples = mu[material_ids] + sigma[material_ids].*z_samples;
  vector[N_holders] mu_holders = mu_samples[sample_ids] + sigma_sample[material_ids][sample_ids].*z_holders;
  vector[N_ids] mu_ids = mu_holders[holder_ids][location_ids] + sigma_holders[material_ids][sample_ids][holder_ids][location_ids].*z_measurements;
}
model{

  mu ~ std_normal();
  mu_sigma ~  normal(0,1); 

  sigma ~ std_normal();
  sigma_sample ~ std_normal();
  sigma_holders ~ std_normal();
  sigma_materials ~ normal(0,1);

  z_samples ~ std_normal();
  z_holders ~ std_normal();
  z_measurements ~ std_normal();

  sigma_common ~ inv_gamma(mu_sigma,sigma_materials);

  y[train_ids] ~ normal(mu_ids[track_ids][train_ids], sigma_common[material_sigma_ids][train_ids]); // 
}
generated quantities {
 vector[N] log_likelihood;
  array[N] real y_hat;
  //array[N_ids] real y_hat2;

 for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu_ids[track_ids[i]],sigma_common[material_sigma_ids[i]]);
 } 
 y_hat = normal_rng(mu_ids[track_ids],sigma_common[material_sigma_ids]);
 //y_hat2 = normal_rng(mu_ids,sigma_common[material_sigma_ids]);
}