# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:14:46 2022

@author: Thomas Sear
Code to model simple harmonic motion using a physics informed neural network
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch

st = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name())

#Class to save the best model from training
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            #print(f"\nBest validation loss: {self.best_valid_loss}")
            #print(f"\nSaving best model for epoch: {epoch+1}\n")
            # torch.save({
            #     'epoch': epoch+1,
            #     'model_strtate_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': criterion,
            #     }, 'outputs/best_model.pth')
            torch.save(model,'outputs/best_model.pth' )

t = torch.linspace(0,100,1001) # time
A = 20  # Amplitude
k = 10  # Spring Constanc
m = 100 # Mass

omega = np.sqrt(k/m) # Angular Frequency
size_sample = 5        # How many points to sample

def SHM_analytic(t, A=A, omega=omega): 
    #Function to find dispacement 
    x = A*torch.cos(omega*t)
    return x

x = SHM_analytic(t)

torch.manual_seed(100) #Set random seed

t_data = 100*torch.rand((size_sample,1))    #Get random training data
x_data = SHM_analytic(t_data)               #Get displacement training data

#Add noise
#noise = np.random.normal(0,2,size=size_sample)
#x_data = x_data+noise

#Plot Training Data
plt.figure(1)
fig, ax = plt.subplots()
ax.plot(t,x, label='Analytical Solution')
ax.scatter(t_data,x_data, c='r', label='Randomly Sampled Points')
plt.xlabel("Time")
plt.ylabel("Displacement")
legend = fig.legend()



#Neural network 
class SHMmodel1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, n_layers):
        super().__init__()
        activation = nn.Softplus #(Non-Linear Activation)
        self.in_layer = nn.Sequential(*[
            nn.Linear(input_features, hidden_units), activation()]) #Define initial layer
        
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(*[nn.Linear(hidden_units, hidden_units), activation()]) 
            for _ in range(n_layers-1)]) #Define hidden layers
        
        self.out_layer = nn.Linear(hidden_units, output_features) #Define output layer
        
        self.apply(self._init_weights) #Initialise Weights?
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.2)
            # if module.bias is not None:
            #     module.bias.data.zero_()

        
    def forward(self, x):
        x = self.in_layer(x) #Put input through first layer
        x = self.hidden_layers(x) #Put x thorugh hidden layers
        x = self.out_layer(x) #Put x thorugh output layer
        return x

    
model = SHMmodel1(1,1,50,5)    


xx = torch.linspace(0,100,100).reshape(100,1)
yy = model(xx)

plt.figure(2)
fig, ax = plt.subplots()
plt.plot(xx.detach().numpy(),yy.detach().numpy())
plt.title("Before training")

#Create optimiser 

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

#Tracked parameters
total_loss = []
epoch_count = []
phys_loss = []
data_loss = []

n_phys = 1000
#xx_phys = torch.linspace(0,100,n_phys,requires_grad=True).reshape((n_phys),1)
#xx_phys = 100*torch.rand((n_phys,1),requires_grad=True)

alpha = 0.05   #0.05
#alpha = 1
loss_fn = nn.MSELoss() 
save = SaveBestModel()

t_phys = torch.linspace(0,100,n_phys,requires_grad=True).reshape((n_phys),1)
t_phys = t_phys.to(device)

model=model.to(device)
x_data = x_data.to(device)
t_data = t_data.to(device)

for i in range(2): # reducing learning rate after each
    for epoch in range(30000):
        optimiser.zero_grad()
        
        
        xx_phys = model(t_phys)
        dt, = torch.autograd.grad(xx_phys,t_phys,torch.ones_like(xx_phys),create_graph=True)
        dt2, = torch.autograd.grad(dt,t_phys,torch.ones_like(dt),create_graph=True)
        
        eqn = dt2+(k/m)*xx_phys #xx_phys                          
        loss_phys = (eqn**2).mean()
        
        loss_data = ((x_data - model(t_data))**2).mean()
        
        #loss_bc = 
        
        loss = alpha*loss_data + (1-alpha)*loss_phys
        if epoch%10==0:
            epoch_count.append(epoch)
            total_loss.append(loss.detach().to('cpu').numpy())
            phys_loss.append(loss_phys.detach().to('cpu').numpy())
            data_loss.append(loss_data.detach().to('cpu').numpy())
        if epoch%1000==0:
            print(f"Epoch: {epoch} | Physics Loss: {(1-alpha)*loss_phys} | Data Loss: {alpha*loss_data} | total loss: {loss}")
        
        save(loss, epoch, model, optimiser, loss_fn)    
        loss.backward()
        optimiser.step()
    # Drop learning rate by half
    for grp in optimiser.param_groups:
        grp['lr']*=0.5
        print( grp['lr'])

model.to('cpu')
t_pinn = torch.linspace(0,100,100).reshape(100,1)
x_pinn = model(t_pinn)
x_data = x_data.to('cpu')
t_data = t_data.to('cpu')


plt.figure(3)
fig, ax = plt.subplots()
plt.plot(t,x,label='Analytical Solution')
plt.plot(t_pinn.detach().numpy(),x_pinn.detach().numpy(), c='m', label='PINN Solution')
ax.scatter(t_data,x_data, c='r', label='Randomly Sampled Points')
plt.title("After training")
plt.xlabel("Time")
plt.ylabel("Displacement")
legend = fig.legend()
plt.savefig('regular_SHM')

et = time.time()

elapsed_time = et-st
print('Execution time:', elapsed_time, 'Seconds')

plt.show()

#plot loss curve
fig, ax = plt.subplots()
total_loss = np.asarray(total_loss)
phys_loss = np.asarray(phys_loss)
data_loss = np.asarray(data_loss)
x = torch.linspace(0,np.size(phys_loss),np.size(phys_loss))*10

plt.plot(x,phys_loss, label='Physics Loss')
plt.plot(x,data_loss, label='Data Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.yscale('log')
legend = fig.legend()
plt.show()


