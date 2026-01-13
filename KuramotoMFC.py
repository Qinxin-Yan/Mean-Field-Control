import torch
import numpy as np
import matplotlib.pyplot as plt

from solver import BSDEsolver
from solver import bsde


dim_h, N, itr, batch_size = 32, 200, 3001, 6000
#dim_h is the NN's hidden layer's width, N is the total time step, itr is the iteration, batch_size is the number of sample path
kappa = 3
epsilon=0  #common noise
bins=80
T , beta = 10, 1  
W = torch.randn(batch_size, N, requires_grad=False) * np.sqrt(T/N)
#$\beta$ is the discount factor, T is the time truncation

# Define an initial PDF function
#def inverse_cdf_one(x):
#    return 


def inverse_cdf(x):
    if x<=1/2:
        return 0.5*np.pi*x+np.pi/2
    else:
       return 0.5*np.pi*x +np.pi
    
def inverse_cdf_3cluster(x):
    if x<=1/3:
        return 3*np.pi*x/8+7*np.pi/16
    if 1/3<x<=2/3:
        return 3*np.pi*x/8+13*np.pi/16
    if x>2/3:
        return 3*np.pi*x/8+19*np.pi/16
    
#batch_size=2800
#while(batch_size<=4000):
#   count=1
#    loss=[0]*10
#    while count<=10:
#        x=torch.zeros((batch_size,1))
        x1=torch.zeros((batch_size,1))
        for i in range(batch_size):
            x[i,0]=inverse_cdf_3cluster(torch.rand(1))
            x1[i,0]=inverse_cdf(torch.rand(1))
        equation = bsde(x ,T, N, kappa, beta, batch_size,epsilon)
        bsde_solver = BSDEsolver(equation, dim_h)
        loss[count-1]=bsde_solver.train(itr, log=False)
        print(count, batch_size, loss[count-1])
        count+=1
#    print(batch_size,np.mean(loss),np.var(loss))
#    batch_size+=200


#x=intial_distribution.sample(sample_shape=(batch_size,1))
#x=torch.zeros((batch_size,1))
x1=torch.zeros((batch_size,1))

for i in range(batch_size):
    #print(torch.rand(1))
    #x1[i,0]=inverse_cdf_3cluster(torch.rand(1))
    #x1[i,0]=inverse_cdf(torch.rand(1))
    x1[i,0]=torch.rand(1)*2*np.pi

#x1=torch.randn((batch_size,1))*1.5
#print(x1)
#data=x1[:,0]
#data.requires_grad_(False)
#plt.hist(data, bins=50, edgecolor='black')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()





print("kappa=",kappa )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


equation = bsde(x1 ,T, N, kappa, beta, batch_size,epsilon,W)

bsde_solver = BSDEsolver(equation, dim_h)
loss=bsde_solver.train(itr, log=False)
bsde_solver.test(bins)
#bsde_solver.test_control(bins)
#bsde_solver.testsincos_()
#bsde_solver.heatmap_mu(bins,x1)
#bsde_solver.test_control_mu(bins,x1)


