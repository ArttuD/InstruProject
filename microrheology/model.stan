//definition of the logic for the model (in STAN language)
//computation of this model is performed later using stan function

//data pre-processing
// 1. make dataframe of all the data - with all the labels, details
// 2. decide paramaters of interest (G,phi) 
// 3. normalize the parameters of interest - z-score normalizing (all data together for now) - only for model stabilitu
// 4. make a dictionary out of the dataframe, because in C language (which is used for models) it doesnt have dataframe

//dictionary
data
{
  int N; // number of datapoints
  int N_train; //portion training data for model 80%, plot 20%
  int N_samples; 
  int N_holders;
  int N_locations;
  int N_materials;
  int N_ids;
  int material_ids[N_samples];
  int sample_ids[N_holders];
  int holder_ids[N_locations];
  int location_ids[N_ids];
  int track_ids[N];
  real y[N];    //in stan float is real
  int train_ids[N_train];
}

parameters
{
  vector[N_materials] mu;       //prior avg value tau for specific material
  vector<lower=0>[N_materials] sigma; //prior avg std of tau for specific material, constrained to be above 0

  vector<lower=0>[N_materials] sigma_sample; //std is constant for samples within a materials
  vector<lower=0>[N_materials] sigma_holders;

  vector[N_holders] z_holders; //for z-score normalizaiton
  vector[N_samples] z_samples; //for z-score normalizaiton

  vector[N_ids] z_measurements; //for z-score normalizaiton

  real mu_sigma;
  vector<lower=0>[N_materials] sigma_materials;
  vector<lower=0>[N_ids] sigma_common;
}

//relationships (functions) between the priors that mapp levels
transformed parameters {
  vector[N_samples] mu_samples = mu[material_ids] + sigma[material_ids].*z_samples;
  vector[N_holders] mu_holders = mu_samples[sample_ids] + sigma_sample[material_ids][sample_ids].*z_holders;
  vector[N_ids] mu_ids = mu_holders[holder_ids][location_ids] + sigma_holders[material_ids][sample_ids][holder_ids][location_ids].*z_measurements;
}

model
{
  mu ~ std_normal();
  sigma ~ std_normal();
  sigma_sample ~ std_normal();
  sigma_holders ~ std_normal();
  z_samples ~ std_normal();
  z_holders ~ std_normal();
  z_measurements ~ std_normal();

  mu_sigma ~ normal(0,1); //weakly informative, practically non-informative, priors are usually normal distributions with mean zero and std 1
  sigma_materials ~ normal(0,1);

  //inverse gamma distribution, >0, favors values close to 0
  sigma_common ~ inv_gamma(mu_sigma,sigma_materials[material_ids][sample_ids][holder_ids][location_ids]);
  
  // function fitting the y (an individual measurement of the specific bead) - likelihood; contains inside the prior - observation model
  y[train_ids] ~ normal(mu_ids[track_ids][train_ids],sigma_common[track_ids][train_ids]); 
}

//for model diagnostics for the result of the model which evaluates the parameters that have the highest probability
// compares how likely are the values that the model predicts (model predicts the trend) to all data; as R2 in linear regression
generated quantities {
 vector[N] log_likelihood;
 real y_hat[N];
 real y_hat2[N_ids];
 for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu_ids[track_ids[i]],sigma_common[track_ids[i]]);
 } 
 y_hat = normal_rng(mu_ids[track_ids],sigma_common[track_ids]);
 y_hat2 = normal_rng(mu_ids,sigma_common);
}