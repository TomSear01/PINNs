# -*- coding: utf-8 -*-
'''
PINN to locate earthquake.
Uses pretrained PINN for velocity field

'''


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import time
import matplotlib.colors as colors
from scipy import integrate
from scipy import optimize as optim

torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80705)
np.random.seed(100)

#Device Configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
#if device == 'cuda': 
#    print(torch.cuda.get_device_name()) 

device = 'cpu'
#--------------------------- Plotting functions--------

def plotTraining(x,y,v,src,rcvr):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    fig,ax=plt.subplots(1,1)
    cp = ax.pcolor(X,Y, F_xy,cmap="rainbow")
    plt.scatter((src[:,0]),(src[:,1]),marker='^', c='k', linewidths=3)
    plt.scatter(rcvr[:,0],rcvr[:,1],marker ='.',c='k',linewidths=3)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    #ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plotFit(x,y,v):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    fig,ax=plt.subplots(1,1)
    #cp = ax.contourf(X,Y, F_xy,50,cmap="rainbow") # regular contour
    #cp = ax.contourf(X,Y, F_xy,50,cmap="rainbow",norm=colors.LogNorm(vmin=F_xy.min(), vmax=F_xy.max())) #log contour
    #cp = ax.pcolor(X,Y, F_xy,cmap="rainbow",norm=colors.LogNorm(vmin=F_xy.min(), vmax=F_xy.max())) # log color
    cp = ax.pcolor(X,Y, F_xy,cmap="rainbow")
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('misfit.png')
    plt.show()


def plotResult(x,y,v,src,rcvr,ray):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    rayx = ray[:,0]
    rayy = ray[:,1]
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    plt.scatter((src[:,0],rcvr[:,0]),(src[:,1],rcvr[:,1]),marker='.', c='k', linewidths=3)
    plt.scatter(rayx,rayy, c='k', marker = '.', linewidths = 0.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
class NN(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Softmax()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        
        #self.p = torch.nn.Parameter(torch.rand((6,5))) #x,y,spreadX, spreadY, Amplidtud
        #self.p1 = torch.nn.Parameter(torch.tensor
        #self.p = torch.nn.Parameter(torch.tensor([[0.3,0.5,0.1,0.2],[0.7,0.7,0.1,1],[0.5,0.1,0.09,0.09]])) #True values  x,y,spreadX, spreadY
        self.p = torch.tensor([[0.5,0.85,0.7,0.1],[0.1,0.4,0.2,0.2],[0.8,0.2,0.4,0.4]])
        #self.free_p = torch.nn.Parameter(torch.tensor([0.85]))#
        self.free_p = torch.nn.Parameter(torch.rand(12))
        
        
            
    'foward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)   
        
        X = x[:,0]
        Y = x[:,1]
        
        p1 = 1*torch.exp(-1*(((X-self.free_p[0])**2)/(2*self.free_p[1]**2)+((Y-self.free_p[2])**2)/(2*self.free_p[3]**2)))
        p2 = 1*torch.exp(-1*(((X-self.free_p[4])**2)/(2*self.free_p[5]**2)+((Y-self.free_p[6])**2)/(2*self.free_p[7]**2)))
        p3 = 1*torch.exp(-1*(((X-self.free_p[8])**2)/(2*self.free_p[9]**2)+((Y-self.free_p[10])**2)/(2*self.free_p[11]**2)))

        v = (p1+p2+p3)
        v = v.unsqueeze(1)              
        return v

                        
    def loss(self,x,y):              
        loss = self.loss_function(self.forward(x), y)              
        return loss

        
class pinn(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Softplus()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
        self.layers = layers
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=0.1)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,L):
        
        if torch.is_tensor(L) != True:         
            L = torch.from_numpy(L)                        
        
        #convert to float
        a = L.float()
        
        for i in range(len(PINN_layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)
        
        if L.dim() > 1:
          a = (1-L[:,0].unsqueeze(1))*L[:,1:3]+L[:,0].unsqueeze(1)*L[:,3:5]+L[:,0].unsqueeze(1)*(1-L[:,0].unsqueeze(1))*a #Enforce output to source and reciever
        else:
          a = (1-L[0])*L[1:3]+L[0]*L[3:5]+L[0]*(1-L[0])*a #Enforce output to source and reciever
          
        return a
                        
    def loss_PDE(self, Lambda):                    
        L = Lambda.clone()                  
        L.requires_grad = True    
        rL = self.forward(L) #NN outputs coordinate of ray at point L
              
        vel = vel_model.forward(rL)
        slow = (vel**-1)

        dslow_xy = autograd.grad(slow, rL, torch.ones_like(slow).to(device),retain_graph=True, create_graph=True)[0]
        
        dslow_x = dslow_xy[:,0].unsqueeze(1)
        dslow_y = dslow_xy[:,1].unsqueeze(1)

        rLx = rL[:,0].unsqueeze(1)
        rLy = rL[:,1].unsqueeze(1)

        drLx_L = autograd.grad(rLx, L, torch.ones_like(rLx).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)x coordiante / dLambda
        drLx_L = drLx_L[:,0].unsqueeze(1)
        
        drLy_L = autograd.grad(rLy, L, torch.ones_like(rLy).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)y coordiante / dLambda
        drLy_L = drLy_L[:,0].unsqueeze(1)
        
        mulx = slow*drLx_L 
        muly = slow*drLy_L
        
        
        d_mulx = autograd.grad(mulx, L, torch.ones_like(mulx).to(device), retain_graph=True, create_graph=True)[0]
        d_mulx = d_mulx[:,0].unsqueeze(1)
        
        d_muly = autograd.grad(muly, L, torch.ones_like(muly).to(device), retain_graph=True, create_graph=True)[0]
        d_muly = d_muly[:,0].unsqueeze(1)
        
        eqnx = d_mulx-dslow_x
        eqny = d_muly-dslow_y
        
        eqn = torch.cat((eqnx,eqny),1) # Put x and y equations back into one tensor
       
        
        loss_f = (eqn**2).mean()  #Mean squared         
        return loss_f
        
    def loss_ray(self, Lambda): 
        L = Lambda.clone()                  
        L.requires_grad = True    
        rL = self.forward(L) #NN outputs coordinate of ray at point L
              
        vel = vel_model.forward(rL)
        slow = (vel**-1)

        dslow_xy = autograd.grad(slow, rL, torch.ones_like(slow).to(device),retain_graph=True, create_graph=True)[0]
        
        dslow_x = dslow_xy[:,0].unsqueeze(1)
        dslow_y = dslow_xy[:,1].unsqueeze(1)

        rLx = rL[:,0].unsqueeze(1)
        rLy = rL[:,1].unsqueeze(1)

        drLx_L = autograd.grad(rLx, L, torch.ones_like(rLx).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)x coordiante / dLambda
        drLx_L = drLx_L[:,0].unsqueeze(1)
        
        drLy_L = autograd.grad(rLy, L, torch.ones_like(rLy).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)y coordiante / dLambda
        drLy_L = drLy_L[:,0].unsqueeze(1)
        
        mulx = slow*drLx_L 
        muly = slow*drLy_L
        
        
        d_mulx = autograd.grad(mulx, L, torch.ones_like(mulx).to(device), retain_graph=True, create_graph=True)[0]
        d_mulx = d_mulx[:,0].unsqueeze(1)
        
        d_muly = autograd.grad(muly, L, torch.ones_like(muly).to(device), retain_graph=True, create_graph=True)[0]
        d_muly = d_muly[:,0].unsqueeze(1)
        
        eqnx = d_mulx-dslow_x
        eqny = d_muly-dslow_y
        
        eqn = torch.cat((eqnx,eqny),1) # Put x and y equations back into one tensor
        out = torch.mean(eqn, 1, True)
        return out

#Travel time funciton using tensors
def travelTimeT(L):

  L = LambdaSR_T.clone()
  n_lambda = 101
  
    
  LambdaSR = L.to(device)  
  LambdaSR.requires_grad=True
  ray = PINN.forward(LambdaSR)
  
  X_dx = autograd.grad(ray[:,0],LambdaSR,torch.ones_like(ray[:,0]),retain_graph=True)[0]
  X_dx = X_dx[:,0]
  X_dy = autograd.grad(ray[:,1],LambdaSR,torch.ones_like(ray[:,1]),retain_graph=True)[0]
  X_dy = X_dy[:,0]
  X_dL = (X_dx**2 + X_dy**2)**0.5 #Jacobian working now :)
  X_dL = X_dL.unsqueeze(1)
  
  time_guess = torch.zeros(n_rays)
  
  for i in range(n_rays):
    vel = vel_model.forward(ray[i*n_lambda:n_lambda*(i+1),:])
    slow = (vel**-1)
    jac = X_dL[i*n_lambda:n_lambda*(i+1)]
    var = slow*jac
    time_guess[i] = torch.trapezoid(var.squeeze(),LambdaSR[i*n_lambda:n_lambda*(i+1),0])
    
 
  return time_guess

    
#--------------------------------------------------------------------------------------------------------------
#Initialise Neural Networks
#Velocity NN
xUB = 1 #x upper bound
yUB = 1 #y lower bound

x = np.linspace(0,xUB,1001)
y = np.linspace(0,yUB,1001)

x = np.expand_dims(x,1)
y = np.expand_dims(y,1)

# Create the mesh 
X,Y=np.meshgrid(x,y)

#Velocity Field
layers_vel = np.array([2,100,100,100,100,1]) #

vel_model = NN(layers_vel) #Initialise Velocity NN
vel_model.to(device)


xy_stack = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

#Domain Bounds
lb = xy_stack[0]
ub = xy_stack[-1]

xy_stack = torch.from_numpy(xy_stack)
xy_stack = xy_stack.to(device)
v = vel_model.forward(xy_stack)


v_plot = v.detach().numpy()
v_plot = np.reshape(v_plot,(1001,1001))

#True velocity
p = np.array([[0.5,0.85,0.7,0.1],[0.1,0.4,0.2,0.2],[0.8,0.2,0.4,0.4]])

p1 = 1*np.exp(-1*(((X-p[0,0])**2)/(2*p[0,2]**2)+((Y-p[0,1])**2)/(2*p[0,3]**2)))
p2 = 1*np.exp(-1*(((X-p[1,0])**2)/(2*p[1,2]**2)+((Y-p[1,1])**2)/(2*p[1,3]**2)))
p3 = 1*np.exp(-1*(((X-p[2,0])**2)/(2*p[2,2]**2)+((Y-p[2,1])**2)/(2*p[2,3]**2)))


v = (p1+p2+p3)

plotFit(x,y,v)
#plotFit(x,y,v-v_plot)


#Ray Tracing PINN
PINN_layers = np.array([5,100,100,100,100,2]) #8 hidden layers  #8 hidden layers (tried with only 2 hidden layers and didn't work :( )

total_loss = []
epoch_count = []

PINN = pinn(PINN_layers)
PINN = PINN.to(device)

#--------------------------------------------------------------------------------
##Get Synthetic Data
#SOURCES AND RECIEVERS
n_src = 2
n_rays = 15*n_src

src = np.zeros((n_rays,2))
rcvr = np.ones((n_rays,2))

src_loc = np.array([[0.8,0.075],[0.4,0.4]])#,[0.2,0.2],[0.7,0.7],[0.3,0.8]])


###########
for i in range(n_src): #Fill source array
  src[i*15:(i+1)*15,0] =  src_loc[i,0]
  src[i*15:(i+1)*15,1] =  src_loc[i,1]
  
  rcvr[i*15:((i+1)*15)-10,0] = np.linspace(0.05,0.95,5)
  rcvr[((i+1)*15)-10:((i+1)*15)-5,1] = np.linspace(0.1,0.9,5)
  rcvr[((i+1)*15)-10:((i+1)*15)-5,0] = 0
  rcvr[((i+1)*15)-5:((i+1)*15),1] = np.linspace(0.1,0.9,5)


srcT = torch.from_numpy(src).to(device)
rcvrT = torch.from_numpy(rcvr).to(device)

plotTraining(x,y,v,src,rcvr)
#Ray
#Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y

LambdaSR = np.zeros((n_rays*101,5))
for i in range(n_rays):
    LambdaSR[i*101:101*(i+1),0] = np.linspace(0,1,101)
    LambdaSR[i*101:101*(i+1),1:3] = src[i,:]
    LambdaSR[i*101:101*(i+1),3:5] = rcvr[i,:]
    
    
LambdaSR_T = torch.from_numpy(LambdaSR)
LambdaSR_T = LambdaSR_T.to(device)

##########################################
ray = PINN.forward(LambdaSR_T).detach().to('cpu').numpy()

#plotResult(x,y,v_plot,src,rcvr,ray)

travel_time = np.load('time_real.npy') #Load synthetic travel times
travel_time = torch.from_numpy(travel_time)
#--------------------------------------------------------------------------------------
phys_loss = []
data_loss = []
total_loss = []
epoch_count = []

lr_PINN = 1e-3
optimiser_PINN = torch.optim.Adam(PINN.parameters(),lr=lr_PINN,amsgrad=False)

lr_dual = 1e-2
params = list(vel_model.parameters()) + list(PINN.parameters())
optimiser_dual = torch.optim.Adam(params,lr=lr_dual,amsgrad=False)
loss_PINN = 7
epoch = 0
#Train Ray tracer in random velocity NN
for j in range(100000):
  if j == 0:


#    checkpoint = torch.load('PINN_trained2.pth')
#    PINN.load_state_dict(checkpoint['model_state_dict'])
#    optimiser_PINN.load_state_dict(checkpoint['optimiser_state_dict'])#

#    PINN.train()
#    ray = PINN.forward(LambdaSR_T).detach().to('cpu').numpy()

#    plotResult(x,y,v_plot,src,rcvr,ray)
    loop = 30000
    loop1 = 1
    for i in range(loop1): # reducing learning rate after each
      print(f"\nLoop: {i+1}, lr:{lr_PINN}")
      for epoch in range(loop):#

        optimiser_PINN.zero_grad()
        Lambda_IN = LambdaSR_T
        
        loss_PINN = PINN.loss_PDE(Lambda_IN)
        
        if epoch%1000==0:
            epoch_count.append(epoch)
            total_loss.append(loss_PINN.detach().to('cpu').numpy())
            print(f"Epoch: {epoch} |total loss: {loss_PINN}")
       
        loss_PINN.backward()
        optimiser_PINN.step()
    if lr_PINN > 1e-4 and j==0:# and i > 1:    #1e-5
      for grp in optimiser_PINN.param_groups:
        grp['lr']*=0.1
        lr_PINN = grp['lr']
  torch.save({'epoch': epoch,'model_state_dict': PINN.state_dict(),'optimiser_state_dict': optimiser_PINN.state_dict(),'loss':loss_PINN}, 'PINN_trained2.pth')
      
  optimiser_dual.zero_grad()
    
  loss_PINN = PINN.loss_PDE(LambdaSR_T)


  time_guess = travelTimeT(LambdaSR_T)

  loss_data = ((travel_time-time_guess)**2).mean()
  loss = loss_PINN + loss_data #Combine losses
  if j%1000==0:
    print(f"Epoch: {j} | Physics Loss: {loss_PINN} | Travel Time loss: {loss_data} | Total Loss: {loss}")
  if j%100==0:
    phys_loss.append(loss_PINN.detach().to('cpu').numpy())
    data_loss.append(loss_data.detach().to('cpu').numpy())
    total_loss.append(loss.detach().to('cpu').numpy())
  
  loss.backward()
  optimiser_dual.step()
  
  if lr_dual>=1e-4 and j%20000==0:
      for grp in optimiser_dual.param_groups:
        grp['lr']*=0.1
        lr_dual = grp['lr']   



vel = vel_model.forward(xy_stack)


v_plot = vel.detach().numpy()
v_plot = np.reshape(v_plot,(1001,1001))
ray = PINN.forward(LambdaSR_T).detach().to('cpu').numpy()

plotResult(x,y,v_plot,src,rcvr,ray)

plotFit(x,y,v_plot)
plotFit(x,y,v-v_plot)

  














