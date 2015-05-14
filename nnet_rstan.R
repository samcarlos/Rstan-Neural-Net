##nnet in rstan

iris1=iris[1:100,]
random.samp=sample(100,20)

x=iris1[-random.samp,1:4]
x.new=iris1[random.samp,1:4]

y=as.integer(iris1[-random.samp,5])-1




model="
data {
int<lower=0> P;
int<lower=0> N;
int<lower=0> N_new;

int<lower=0> num_nodes;
int<lower=0> num_hidden;

matrix[N,P] x_data;
matrix[N_new,P] new_x_data;

int y_data[N];

}
parameters {
matrix[num_nodes,P] beta_initial;
matrix[num_nodes,num_nodes] beta_middle[num_hidden];
vector[num_nodes] beta_final;

}
transformed parameters{
matrix[N,num_nodes] fitted_middle[num_hidden];

for(n in 1:N)
  for(j in 1:num_nodes) fitted_middle[1,n, j] <- 1/(1+exp(-beta_initial[j]*x_data[n]' ));

for(q in 2:num_hidden)
  for(n in 1:N)
    for(j in 1:num_nodes) fitted_middle[q,n, j] <- 1/ (1+exp(-beta_middle[q, j] * fitted_middle[q-1,n]'));


}

model{

for(n in 1:N)
  y_data[n] ~ bernoulli_logit(beta_final' * fitted_middle[num_hidden,n]');


}
generated quantities{
vector[N_new] predicted_y;
matrix[N_new,num_nodes] new_fitted_middle[num_hidden];

for(n in 1:N_new)
  for(j in 1:num_nodes) new_fitted_middle[1,n, j] <- 1/(1+exp(-beta_initial[j]*new_x_data[n]' ));

for(q in 2:num_hidden)
  for(n in 1:N_new)
    for(j in 1:num_nodes) new_fitted_middle[q,n, j] <- 1/ (1+exp(-beta_middle[q, j] * new_fitted_middle[q-1,n]'));

for(n in 1:N_new)
  predicted_y[n]<-beta_final' * new_fitted_middle[num_hidden,n]';


}


"



stan.dat=list(x_data=scale(x), y_data=y, num_nodes=10,num_hidden=3,P=4,N=80, N_new=20,new_x_data=scale(x.new))
library(rstan)
fit <- stan(model_code = model, data = stan.dat, iter = 1000, chains = 3, verbose = TRUE)
fitmat=as.matrix(fit)



fitted=fitmat[,grep("predicted_y", colnames(fitmat) )]
parameters=fitmat[,grep("beta", colnames(fit))]
