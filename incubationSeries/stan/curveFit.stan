functions {
   /* ... function declarations and definitions ... */
   vector func(vector x, real a, real b, real c, real d){
      return a + b./(1+exp(-(x-c)./d));
   }

}
data {
    // counts
   int N;
   int N_hat;

   // data
   vector[N] time;
   vector[N] area;
   // indicator vectors

   vector[N_hat] x_hat;
}

//,a,k,c,q,b,v
parameters {
   //real<lower = 0> mu_std; 
   real<lower = 0> sigma_std;
   real a;
   real<lower = 0> b;
   real<lower = 0> c;
   real<lower = 0> d;

}
transformed parameters {
   vector[N] slope = func(time, a, b, c, d);
}
model {
   /* ... declarations ... statements ... */

   sigma_std ~ normal(0,2);
   a ~ normal(0,1);
   b ~ normal(1,1);
   c ~ normal(0,1);
   d ~ normal(0,1);

   area ~ normal(slope, sigma_std);
}
generated quantities {
   vector[N] log_lik;
   real y_hat[N_hat];

   for (i in 1:N){
      log_lik[i] = normal_lpdf(area[i] | slope[i], sigma_std);
   }
   y_hat = normal_rng(func(x_hat, a, b, c, d), sigma_std);

}

