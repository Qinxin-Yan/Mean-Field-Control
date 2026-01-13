import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Basis=10


class bsde():
    def __init__(self, x, T, N, kappa, beta, batch_size,epsilon, W):
        self.x=x
        self.T = T
        self.N=N
        self.kappa=kappa
        self.beta=beta
        self.batch_size=batch_size
        self.epsilon=epsilon
        self.W=W

class PeriodicActivationLayer(nn.Module):
    def __init__(self):
        super(PeriodicActivationLayer, self).__init__()
        # Define a, b, and c as learnable parameters
        self.a = nn.Parameter(torch.Tensor(1))
        self.b = nn.Parameter(torch.Tensor(1))
        self.c = nn.Parameter(torch.Tensor(1))

        # Initialize the parameters (you can choose different initialization methods)
        nn.init.uniform_(self.a)
        nn.init.uniform_(self.b)
        nn.init.uniform_(self.c)

    def forward(self, x):
        return self.a * torch.cos(x) + self.b * torch.sin(x) + self.c



class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.periodic_activation = PeriodicActivationLayer()
        self.linearh1 = nn.Linear(2*Basis+1, dim_h)
        #self.linearh1 = nn.Linear(1, dim_h).requires_grad_(False)
        self.linearh2 = nn.Linear(dim_h, dim_h)
        self.linearh3 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, 1)
        self.equation = equation

    def phi(self, x):
        #one hidden layer, fully-connected NN
        #x = self.periodic_activation(x)
        x = F.relu(self.linearh1(x))
        #x = F.relu(self.linearh2(x))
        x = F.relu(self.linearh3(x))
        return self.linear3(x)

    def forward(self):
        N = self.equation.N
        kappa=self.equation.kappa
        eps=self.equation.epsilon
        delta_t = self.equation.T / N
        batch_size = self.equation.batch_size
        W=self.equation.W
        #W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
        W_0= torch.randn(batch_size, 1, device=device, requires_grad=False) * np.sqrt(delta_t)
        x = self.equation.x
        #print(x)
        tmp = torch.zeros(2*Basis, device=device, requires_grad=False)  # Move tmp to the same device as x
        alpha = torch.zeros(batch_size, device=device)  # Move alpha to the same device as x
        J = 0  #J is the cost function
        for i in range(N):
            #print(i,x)
            
            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            x=x.squeeze(0)
            #for k in range(Basis): #approximate distribution at time t, with fourier series
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
            input=torch.cat([x,tmp.expand(batch_size,-1)],1)
            #input=x
            #print(i,x,tmp, input)
            alpha=self.phi(input)
            #print(i,alpha)
            w = W[:,i].reshape(batch_size,-1)
            #linear quadratic
            #J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 + kappa*( torch.mean(torch.pow(x,2)) - torch.mean(x) ** 2) )
            #Kuramoto
            J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 - kappa*( torch.mean(torch.cos(x)) ** 2 + torch.mean(torch.sin(x)) ** 2) / 2)
            #x = (x + alpha * delta_t + w)%(2*np.pi)-np.pi*torch.ones((batch_size,1))
            x = (x + alpha * delta_t + w + eps * W_0)%(2*np.pi)
            #x = (x + alpha * delta_t + w + eps * W_0)
        J = J * delta_t
        return x,J

class BSDEsolver():
    def __init__(self, equation, dim_h):
        self.model = Model(equation, dim_h).to(device)
        self.equation = equation
        self.batch_size=equation.batch_size

    def train(self, itr, log):
        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_data = []
        samples = []

        for i in range(itr):
            
            optimizer.zero_grad()
            x,J = self.model()
            #print(J.item())
            #J_target = torch.zeros(J.shape, device=device)  
            loss = J  # Pass J and J_target to the loss function
            loss_data.append(float(loss))
            #samples.append(np.histogram(x[:,0].detach().numpy(),bins=5,range=(-np.pi,np.pi),density=True)[0])
            #print(np.histogram(x[:,0].detach().numpy(),bins=5,density=True)[1])
            loss.backward()
            optimizer.step()
            #print(i, loss.item())
            if i%500==0:
                print(i, loss.item())
        if log:
            np.save('loss_data', loss_data)
        return loss.item()
        #data=np.array(samples)
        #print(data)
        #sns.heatmap(data, cmap="YlGnBu")
        #plt.show()

    def test(self,num_bin):
        samples1=[]
        samples = []
        x=self.equation.x
        x1=self.equation.x
        #print("initial:",x)
        samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
        samples.append(np.histogram(x1[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
        #print(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-np.pi,np.pi),density=True)[1])
        T=self.equation.T
        N=self.equation.N
        eps=self.equation.epsilon
        kappa=self.equation.kappa
        batch_size = self.equation.batch_size
        delta_t=T/N
        W1=self.equation.W
        W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
        W_0=torch.randn(batch_size,1,device=device, requires_grad=False)*np.sqrt(delta_t)      #common noise
        tmp = torch.zeros(2*Basis, device=device, requires_grad=False)
        alpha = torch.zeros(batch_size, device=device)
        J=0
        for i in range(N):
            #samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-3,3),density=True)[0])
            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            x=x.squeeze(0)
            #print(i,x)
            #for k in range(Basis): #approximate distribution at time t, with fourier series
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
            input=torch.cat((x,tmp.expand(batch_size,-1)),1)
            #input=x
            #print(i,x,tmp, input)
            alpha=self.model.phi(input)
            #print(i,alpha)
            w = W1[:,i].reshape(batch_size,-1)
            #J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 + kappa*( torch.mean(torch.pow(x,2)) - torch.mean(x) ** 2) )
            J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 - kappa*( torch.mean(torch.cos(x)) ** 2 + torch.mean(torch.sin(x)) ** 2) / 2)
            #x = (x + alpha * delta_t + w)%(2*np.pi)-np.pi*torch.ones((batch_size,1))
            x = (x + alpha * delta_t + w + eps*W_0)%(2*np.pi)
            #x = (x + alpha * delta_t + w + eps * W_0)
            #print(i,x)
            samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
            if i==np.floor(N/5):
                x1=x
        data=np.array(samples)
        J=J*delta_t
        print("test:",J.item())
        #print(data)
        #print(data.shape)
        #x_range=np.linspace(-np.pi,np.pi,num_bin)
        #y_range=np.linspace(0,T,data.shape[1]+1)
        heatmap=sns.heatmap(data, cmap="YlGnBu")
        heatmap.set_xticks([0,int(num_bin/2), num_bin])
        heatmap.set_xticklabels([0, 3.14, 6.28])
        heatmap.set_yticks([0,int(N/2), N])
        heatmap.set_yticklabels([0, T/2, T])
        #heatmap.set_xticklabels(x_range)
        #heatmap.set_yticklabels(y_range)
        plt.xlabel("X")
        plt.ylabel("Time")
        plt.savefig(f"heatmap_kappa{kappa}.png")
        return x1
    

    def test_control(self,num_bin): 
    #plot the control as a function a x, with mu fixed.
        samples = []
        x=self.equation.x
        #print("initial:",x)
        samples.append(np.histogram(x[:,0].detach().numpy(),bins=batch_size,range=(0,2*np.pi),density=True)[0])
        #print(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-np.pi,np.pi),density=True)[1])
        T=self.equation.T
        N=self.equation.N
        eps=self.equation.epsilon
        kappa=self.equation.kappa
        batch_size = self.equation.batch_size
        delta_t=T/N
        W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
        W_0 = torch.randn(batch_size, 1, device=device, requires_grad=False) * np.sqrt(delta_t)
        tmp = torch.zeros(2*Basis, device=device, requires_grad=False)
        alpha = torch.zeros(batch_size, device=device)
        for i in range(N):
            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            x=x.squeeze(0)
            #for k in range(Basis): #approximate distribution at time t, with fourier series
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
            input=torch.cat((x,tmp.expand(batch_size,-1)),1)
            #input=x
            alpha=self.model.phi(input)
            w = W[:,i].reshape(batch_size,-1)
            x = (x + alpha * delta_t + w + eps*W_0)%(2*np.pi)

        x_test=torch.linspace(0,2*np.pi,num_bin).reshape(num_bin,-1)

        #compute empirical distribution(for the optimal flow) at terminal time T, with fourier basis
        #for k in range(Basis):
        #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
        #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
        x=x.unsqueeze(0)
        k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
        tmp1=torch.cos(k1*x).mean(1).squeeze()
        tmp2=torch.sin(k1*x).mean(1).squeeze()
        tmp[::2]=tmp1
        tmp[1::2]=tmp2
        tmp=tmp*2*np.pi
        x=x.squeeze(0)

        #tmp=tmp*2*np.pi
        #input=torch.cat((x_test,tmp.expand(num_bin,-1)),1)
        input=x_test
        alpha_T=self.model.phi(input).reshape(batch_size).detach().numpy()
        
        #plot
        x=np.linspace(-4,4,batch_size)
        fig,ax = plt.subplots()
        ax.plot(x, alpha_T, label='alpha', marker=',', linestyle='--')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        plt.savefig(f"control_kappa{kappa}.png")



    def heatmap_mu(self,num_bin,x): 
    #with trained NN, plot the heat map with initial x sampled from mu
        samples = []
        #x=self.equation.x
        #print("initial:",x)
        samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
        #print(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-np.pi,np.pi),density=True)[1])
        T=self.equation.T
        N=self.equation.N
        eps=self.equation.epsilon
        J=0
        kappa=self.equation.kappa
        batch_size = self.equation.batch_size
        delta_t=T/N
        W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
        W_0 = torch.randn(batch_size, 1, device=device, requires_grad=False) * np.sqrt(delta_t)
        tmp = torch.zeros(2*Basis, device=device, requires_grad=False)
        alpha = torch.zeros(batch_size, device=device)
        for i in range(N):
            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            x=x.squeeze(0)
            #for k in range(Basis): #approximate distribution at time t, with fourier series
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
            #input=torch.cat((x,tmp.expand(batch_size,-1)),1)
            input=x
            alpha=self.model.phi(input)
            w = W[:,i].reshape(batch_size,-1)
            J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 + kappa*( torch.mean(torch.pow(x,2)) - torch.mean(x) ** 2) )
            #J = J+ np.exp(-self.equation.beta * i * delta_t) * (torch.mean(torch.pow(alpha,2))/2 - kappa*( torch.mean(torch.cos(x)) ** 2 + torch.mean(torch.sin(x)) ** 2) / 2)
            x = (x + alpha * delta_t + w + eps*W_0)%(2*np.pi)
            samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
        J = J * delta_t
        print("J: ",J)
        data=np.array(samples)
        heatmap=sns.heatmap(data, cmap="YlGnBu")
        heatmap.set_xticks([0,num_bin/2,num_bin])
        heatmap.set_xticklabels([0,"pi","2*pi"])
        #heatmap.set_xticklabels(x_range)
        #heatmap.set_yticklabels(y_range)
        plt.xlabel("X")
        plt.ylabel("Time")
        plt.savefig(f"heatmap_kappa_uni{kappa}.png")

    def test_control_mu(self,num_bin,x): 
        #plot the control as a function a x, with mu fixed.
            samples = []
            #x=self.equation.x
            #print("initial:",x)
            samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
            #print(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-np.pi,np.pi),density=True)[1])
            T=self.equation.T
            N=self.equation.N
            eps=self.equation.epsilon
            kappa=self.equation.kappa
            batch_size = self.equation.batch_size
            delta_t=T/N
            W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
            W_0 = torch.randn(batch_size, 1, device=device, requires_grad=False) * np.sqrt(delta_t)
            tmp = torch.zeros(2*Basis, device=device, requires_grad=False)
            alpha = torch.zeros(batch_size, device=device)
            for i in range(N):
                x=x.unsqueeze(0)
                k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
                tmp1=torch.cos(k1*x).mean(1).squeeze()
                tmp2=torch.sin(k1*x).mean(1).squeeze()
                tmp[::2]=tmp1
                tmp[1::2]=tmp2
                tmp=tmp*2*np.pi
                x=x.squeeze(0)
                #for k in range(Basis): #approximate distribution at time t, with fourier series
                #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
                #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
                input=torch.cat((x,tmp.expand(batch_size,-1)),1)
                #input=x
                alpha=self.model.phi(input)
                w = W[:,i].reshape(batch_size,-1)
                x = (x + alpha * delta_t + w + eps*W_0)%(2*np.pi)

            x_test=torch.linspace(0,2*np.pi,num_bin).reshape(num_bin,-1)

            #compute empirical distribution(for the optimal flow) at terminal time T, with fourier basis
            #for k in range(Basis):
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))

            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            input=torch.cat((x_test,tmp.expand(num_bin,-1)),1)
            #input=x
            alpha_T=self.model.phi(input).reshape(num_bin).detach().numpy()
            
            #plot
            x=np.linspace(0,2*np.pi,num_bin)
            fig,ax = plt.subplots()
            ax.plot(x, alpha_T, label='alpha', marker=',', linestyle='--')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.legend()
            plt.savefig(f"control_kappa_mu{kappa}.png")

    def testsincos_(self):
        samples_sin = []
        samples_cos = []
        x=self.equation.x
        samples_sin.append(torch.mean(torch.sin(x)))
        samples_cos.append(torch.mean(torch.cos(x)))
        #print("initial:",x)
        #samples.append(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(0,2*np.pi),density=True)[0])
        #print(np.histogram(x[:,0].detach().numpy(),bins=num_bin,range=(-np.pi,np.pi),density=True)[1])
        T=self.equation.T
        N=self.equation.N
        eps=self.equation.epsilon
        kappa=self.equation.kappa
        batch_size = self.equation.batch_size
        delta_t=T/N
        W = torch.randn(batch_size, N, device=device, requires_grad=False) * np.sqrt(delta_t)
        W_0 = torch.randn(batch_size, 1, device=device, requires_grad=False) * np.sqrt(delta_t)
        tmp = torch.zeros(2*Basis, device=device, requires_grad=False)
        alpha = torch.zeros(batch_size, device=device)
        for i in range(N):
            x=x.unsqueeze(0)
            k1=torch.arange(Basis).unsqueeze(1).unsqueeze(2)+1
            tmp1=torch.cos(k1*x).mean(1).squeeze()
            tmp2=torch.sin(k1*x).mean(1).squeeze()
            tmp[::2]=tmp1
            tmp[1::2]=tmp2
            tmp=tmp*2*np.pi
            x=x.squeeze(0)
            #for k in range(Basis): #approximate distribution at time t, with fourier series
            #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
            #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
            input=torch.cat((x,tmp.expand(batch_size,-1)),1)
            #input=x
            alpha=self.model.phi(input)
            w = W[:,i].reshape(batch_size,-1)
            x = (x + alpha * delta_t + w + eps*W_0)%(2*np.pi)
            samples_sin.append(torch.mean(torch.sin(x)))
            samples_cos.append(torch.mean(torch.cos(x)))


        #compute empirical distribution(for the optimal flow) at terminal time T, with fourier basis
        #for k in range(Basis):
        #    tmp[2*k] = torch.mean(torch.cos((k+1)*x))
        #    tmp[2*k+1]=torch.mean(torch.sin((k+1)*x))
        
        
        #plot
        x=np.linspace(0,T,N)
        fig,ax = plt.subplots()
        ax.plot(x, np.array(samples_cos.detach().numpy()), label='cos', marker=',', linestyle='--')
        ax.plot(x, np.array(samples_sin.detach().numpy()), label='sin', marker='.', linestyle='-')
        ax.set_xlabel('q')
        ax.set_ylabel('v_prime')
        ax.legend()
        plt.savefig(f"sincos{kappa}.png")

