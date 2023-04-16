# -*- coding: utf-8 -*-
'''
PINN for seismic ray tracing. 
Velocity field is described by a neural network

'''


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import time
from scipy import integrate
import rayTracer as rt

torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80701)
np.random.seed(80701)

#Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

device = 'cpu'
#--------------------------- Plotting functions--------

def plotTraining(x,y,v,src,rcvr):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    plt.scatter((src[0],rcvr[0]),(src[1],rcvr[1]),marker='o', c='k', linewidths=1.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def plotResult(x,y,v,src,rcvr,ray):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    rayx = ray[:,0]
    rayy = ray[:,1]
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap='RdBu')#cmap="RdBu")
    #plt.scatter((src[0],rcvr[0]),(src[1],rcvr[1])),marker='o', c='k', linewidths=1.5)
    plt.scatter(rayx,rayy, c='k',marker='.', linewidths=0.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    plt.ylim(0,1)
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    


   
#-------------------------------------------------------------------------------
#Domain
xUB = 1 #x upper bound
yUB = 1 #y lower bound

x = np.linspace(0,xUB,1001)
y = np.linspace(0,yUB,1001)

x = np.expand_dims(x,1)
y = np.expand_dims(y,1)

# Create the mesh 
X,Y=np.meshgrid(x,y)


v_plot = (2*(np.tanh(200*X-100) + 0*Y)+2.5)#/10


#Source
src = np.array([0.05,0.05]) #Location of source
rcvr = np.array([0.95,1]) #Location of reciever (now on surface!)

plotTraining(x,y,v_plot,src,rcvr) 

#Ray
Lambda = np.linspace(0,1,101)
LambdaT = torch.from_numpy(Lambda).unsqueeze(1).to(device)


srcT = torch.from_numpy(src).to(device)
rcvrT = torch.from_numpy(rcvr).to(device)


class pinn(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Softplus()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
        self.iter = 0    
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=0.5)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,L):
        
        if torch.is_tensor(L) != True:         
            L = torch.from_numpy(L)                        
        
        #convert to float
        a = L.float()
        
        for i in range(len(layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)
        
        
        a = (1-L)*srcT+L*rcvrT+L*(1-L)*a #Enforce output to source and reciever
        
        return a
                        
    def loss_PDE(self, Lambda):                    
        L = Lambda.clone()                  
        L.requires_grad = True    
        rL = self.forward(L) #NN outputs coordinate of ray at point L
              
        vel = (2*(torch.tanh(200*rL[:,0]-100) + 0*rL[:,1])+2.5).unsqueeze(1)#/10
        
        slow = (vel**-1)

        dslow_xy = autograd.grad(slow, rL, torch.ones_like(slow).to(device),retain_graph=True, create_graph=True)[0]
        
        dslow_x = dslow_xy[:,0].unsqueeze(1)
        dslow_y = dslow_xy[:,1].unsqueeze(1)

        rLx = rL[:,0].unsqueeze(1)
        rLy = rL[:,1].unsqueeze(1)

        drLx_L = autograd.grad(rLx, L, torch.ones_like(rLx).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)x coordiante / dLambda
        drLy_L = autograd.grad(rLy, L, torch.ones_like(rLy).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)y coordiante / dLambda
        
        mulx = slow*drLx_L 
        muly = slow*drLy_L
        
        
        d_mulx = autograd.grad(mulx, L, torch.ones_like(mulx).to(device), retain_graph=True, create_graph=True)[0]
        d_muly = autograd.grad(muly, L, torch.ones_like(muly).to(device), retain_graph=True, create_graph=True)[0]
        
        eqnx = d_mulx-dslow_x
        eqny = d_muly-dslow_y
        
        eqn = torch.cat((eqnx,eqny),1) # Put x and y equations back into one tensor

        
        
        loss_f = (eqn**2).mean()  #Mean squared         
        return loss_f



lr=1e-3
layers = np.array([1,20,20,20,20,20,20,20,20,20,20,2]) #8 hidden layers (tried with only 2 hidden layers and didn't work :( )

total_loss = []
epoch_count = []

PINN = pinn(layers)
PINN.to(device)


optimiser = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)

#Plot PINN before training
ray = PINN.forward(LambdaT)
ray = ray.to('cpu').detach().numpy()

plotResult(x,y,v_plot,src,rcvr,ray)

num_samples = 25

st = time.time()

for i in range(10): # reducing learning rate after each
    print(f"\nLoop: {i+1}, lr:{lr}")
    for epoch in range(20000):

        optimiser.zero_grad()
        
        #Train eith 20,000 random points each epoch
        #inx = np.random.choice(LambdaT.shape[0],num_samples, replace=False)
        LambdaT_in = LambdaT#[inx]
        
        
        loss = PINN.loss_PDE(LambdaT_in)
         
        if epoch%100==0:
            epoch_count.append(epoch)
            total_loss.append(loss.detach().to('cpu').numpy())
        if epoch%1000==0:
            print(f"Epoch: {epoch} |total loss: {loss}")
       
        loss.backward()
        optimiser.step()

        
    if lr >= 1e-5 and i%3==0:# and i > 1:    #1e-5
      for grp in optimiser.param_groups:
        grp['lr']*=0.1
        lr = grp['lr']
    
    
et = time.time()

elapsed_time = et-st
print('Execution time:', elapsed_time, 'Seconds')


Lambda_small = np.linspace(0,1,15)
LambdaT = torch.from_numpy(Lambda_small).unsqueeze(1).to(device)
ray = PINN.forward(LambdaT)
ray = ray.to('cpu').detach().numpy()

plotResult(x,y,v_plot,src,rcvr,ray)

device = 'cpu'

PINN= PINN.to('cpu')
srcT = torch.from_numpy(src).to('cpu')
rcvrT = torch.from_numpy(rcvr).to('cpu')

#Travel Time
#def slow(L):
#  L = np.array([L])
#  L = torch.from_numpy(L)
#  L.requires_grad=True
#  ray = PINN.forward(L)
#  X_dL = autograd.grad(ray,L,torch.ones_like(ray))[0][0] #This is not doing what I want...
#  v = vel_model.forward(ray).detach().numpy()
#  v = 0*ray[0] + 0*ray[1] +2
#  s = v**-1
#  out = s*X_dL.detach().numpy()
#  return out
  
def slow(L):
  #Function for travel time calculation using scipy.integrate.quad
  #Takes single L value, source and reciever.
  L = torch.from_numpy(np.array([L]))
  L.requires_grad = True
  
  ray = PINN.forward(L)
  
  X_dx = autograd.grad(ray[0],L,torch.ones_like(ray[0]),retain_graph=True)[0][0]
  X_dy = autograd.grad(ray[1],L,torch.ones_like(ray[1]))[0][0]
  X_dL = (X_dx**2 + X_dy**2)**0.5
  #X_dL = autograd.grad(ray,L,torch.ones_like(ray))[0][0]#Jacobian
  v = (2*(torch.tanh(200*ray[0]-100) + 0*ray[1])+2.5).detach().numpy()
  #v = vel_model.forward(ray).detach().numpy()
  s = v**-1
  out = s*X_dL.detach().numpy()
  return out

time = integrate.quad(slow,0.,1.)
print(f'PINN Travel time:{time[0]}')

####Ray tracer
m = np.array([[0.5,0.5],
              [4.5,4.5]])
g=rt.gridModel(m)

mp = g.getVelocity()
g.setVelocity(mp)
g.getSlowness()

#src = np.array([0.05,0.05])
g.shootInitial(src)

#rec1 = np.array([0.8,1])
rec1 = rcvr
travelTimes1, paths1, sensitivities1 = g.findPath(rec1)


allPaths = paths1 
#rt.displayModel(g.getVelocity(),allPaths,cmap=plt.cm.RdBu)
print(f'Ray tracer Travel Time:{travelTimes1}')
#Plot up
x_ray = allPaths[0][:,0]
y_ray = allPaths[0][:,1]

x_plot =x.squeeze(1) 
y_plot =y.squeeze(1)
X,Y= np.meshgrid(x_plot,y_plot)
F_xy = v_plot
rayx = ray[:,0]
rayy = ray[:,1]
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X,Y, F_xy,20,cmap='RdBu')#cmap="RdBu")
#plt.scatter((src[0],rcvr[0]),(src[1],rcvr[1])),marker='o', c='k', linewidths=1.5)
plt.plot(rayx,rayy, c='k', linewidth=1,label = 'PINN Ray')
plt.plot(x_ray,y_ray, c='w', linewidth=1, label = "Snell's Law Ray") 
fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
plt.ylim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.legend()
plt.show()


#Plot loss curve
total_loss = np.asarray(total_loss)
x = np.linspace(0,np.shape(total_loss)[0],np.shape(total_loss)[0])*100
fig,ax=plt.subplots(1,1)
plt.plot(x,total_loss)
plt.yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Iterations')
plt.show()

