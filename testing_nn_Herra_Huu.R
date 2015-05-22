library(rstan)

data = iris[1:100,]
data[,1:4] = scale(data[,1:4])
data[,5] = as.integer(data[,5])-1

N = 80
Nt = nrow(data)-N
train_ind = sample(100,N)
test_ind = setdiff(1:100, train_ind) 

yt = data[test_ind,5]
stan.dat=list(
  num_nodes=10, 
  num_middle_layers=3, 
  d=4, 
  N=N, 
  Nt=Nt, 
  X=data[train_ind,1:4], 
  y=data[train_ind,5],  
  Xt=data[test_ind,1:4])

m <- stan_model("nn.stan")
s <- sampling(m, data = stan.dat, iter = 1000, chains = 4)

fitmat = as.matrix(s)
predictions = fitmat[,grep("predictions", colnames(fitmat))]
parameters = fitmat[,grep("beta", colnames(fitmat))]

mean_predictions = colMeans(predictions)
plot(1:Nt, yt)
lines(1:Nt, mean_predictions, type='p', col='red')
