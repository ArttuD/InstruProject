functions {
   /* ... function declarations and definitions ... */
   vector func(vector x, real a, real b, real c, real d, real q, real v){
      return to_vector(rep_vector(1,size(x)).*(a+b./pow(c+q*exp(-(x-c)./d),1./v)));
   }

}
data {
    // counts
   int N;
   int N_day;
   int N_conc;
   int N_cell_type;
   // data
   vector[N] time;
   vector[N] area;
   // indicator vectors
   int cell_type_indices[N];
   int day_indices[N];
   int conc_indices[N];
}

//,a,k,c,q,b,v
parameters {
   //real<lower = 0> mu_std; 
   real<lower = 0> sigma_std;
   real a;
   real b;
   real c;
   real d;
   real q;
   real<lower=0> v;

}
model {
   /* ... declarations ... statements ... */
   sigma_std ~ std_normal();
   a ~ std_normal();
   b ~ std_normal();
   c ~ std_normal();
   d ~ std_normal();
   q ~ std_normal();
   v ~ std_normal();
   area ~ normal(func(time,  a, b, c, d, q, v), sigma_std);
}

