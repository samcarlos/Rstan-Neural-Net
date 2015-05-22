functions {
  matrix scaled_inv_logit(matrix X) {
    matrix[rows(X), cols(X)] res;
    for(i in 1:rows(X))
      for(j in 1:cols(X)) 
        res[i,j] <- inv_logit(X[i,j])*2-1;
    return res;
  }
  
  vector calculate_alpha(matrix X, vector bias, matrix beta_first, matrix[] beta_middle, vector beta_output) {
    int N;
    int num_nodes;
    int num_layers;
    matrix[rows(X),rows(beta_first)] layer_values[rows(bias)];
    vector[rows(X)] alpha;
    
    N <- rows(X);
    num_nodes <- rows(beta_first);
    num_layers <- rows(bias);
    
    layer_values[1] <- scaled_inv_logit(bias[1] + X * beta_first');   
    for(i in 2:(num_layers-1)) 
      layer_values[i] <- scaled_inv_logit(bias[i] + layer_values[i-1] * beta_middle[i-1]');
    alpha <- bias[num_layers] + layer_values[num_layers-1] * beta_output;

    return alpha;
  }
}
data {
  int<lower=0> N;
  int<lower=0> d;
  int<lower=0> num_nodes;
  int<lower=1> num_middle_layers;
  matrix[N,d] X;
  int y[N];
  int<lower=0> Nt;
  matrix[Nt,d] Xt;
}
transformed data {
  int num_layers;
  num_layers <- num_middle_layers + 2;
}
parameters {
  vector[num_layers] bias;
  matrix[num_nodes,d] beta_first;
  matrix[num_nodes,num_nodes] beta_middle[num_middle_layers];
  vector[num_nodes] beta_output;
}
model{
  vector[N] alpha;
  alpha <- calculate_alpha(X, bias, beta_first, beta_middle, beta_output);
  y ~ bernoulli_logit(alpha);
  
  //priors
  bias ~ normal(0,1);
  to_vector(beta_output) ~ normal(0,1);
  to_vector(beta_first) ~ normal(0,1);
  for(i in 1:(num_middle_layers)) 
    to_vector(beta_middle[i]) ~ normal(0,1);
}
generated quantities{
  vector[Nt] predictions;
  {
    vector[Nt] alpha;
    alpha <- calculate_alpha(Xt, bias, beta_first, beta_middle, beta_output);
    for(i in 1:Nt) 
      predictions[i] <- inv_logit(alpha[i]);
  }
}