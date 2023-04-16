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
import matplotlib.colors as colors

torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80701)
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
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    plt.scatter((src[:,0],rcvr[:,0]),(src[:,1],rcvr[:,1]),marker='.', c='k', linewidths=1.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    ax.set_title('F(x,y)')
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
    vmax = np.max(v)
    vmin = np.min(v)
    cp = ax.pcolor(X,Y, F_xy, vmin=vmin, vmax=vmax,cmap="RdBu")
    fig.colorbar(cp, label='Fit') # Add a colorbar to a plot
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
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
        self.iter = 0    
        self.layers = layers
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=2)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,x):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                        

        
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
                      
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)
        #a = torch.clamp(a,min=0, max = 1.5)        
        return a

                        
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
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=0.5)
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
layers_vel = np.array([2,200,200,200,200,1]) #8 hidden layers

vel_model = NN(layers_vel) #Initialise Velocity NN

vel_model = torch.load('velocity_complex.pth') #Load previously trained velocity NN
vel_model.to(device)
vel_model.eval()

xy_stack = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

#Domain Bounds
lb = xy_stack[0]
ub = xy_stack[-1]

v = vel_model.forward(xy_stack)


v_plot = v.detach().numpy()
v_plot = np.reshape(v_plot,(1001,1001))




#Ray Tracing PINN
PINN_layers = np.array([5,100,100,100,100,100,100,2]) #8 hidden layers  #8 hidden layers (tried with only 2 hidden layers and didn't work :( )

total_loss = []
epoch_count = []

PINN = pinn(PINN_layers)
PINN = torch.load('general_complex.pth') #Is this a generalised model?!
PINN.to(device)


#--------------------------------------------------------------------------------
##Get Synthetic Data
#SOURCES AND RECIEVERS
n_rays = 10

src = np.zeros((n_rays,2))
rcvr = np.ones((n_rays,2))

#Earthquake Source
src[:,0] = 0.3
src[:,1] = 0.2

rcvr[:,0] = np.linspace(0.05,0.95,n_rays) # Equally spaced recievers on the surface

srcT = torch.from_numpy(src).to(device)
rcvrT = torch.from_numpy(rcvr).to(device)

#Ray
#Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y
LambdaSR = np.zeros((n_rays*101,5))
for i in range(n_rays):
    LambdaSR[i*101:101*(i+1),0] = np.linspace(0,1,101)
    LambdaSR[i*101:101*(i+1),1:3] = src[i,:]
    LambdaSR[i*101:101*(i+1),3:5] = rcvr[i,:]
    
    
LambdaSR_T = torch.from_numpy(LambdaSR)
LambdaSR_T = LambdaSR_T.to(device)

ray = PINN.forward(LambdaSR_T).detach().to('cpu').numpy()

plotResult(x,y,v_plot,src,rcvr,ray)



##Travel Time
def slow(L,src,rcvr):
  #Function for travel time calculation using scipy.integrate.quad
  #Takes single L value, source and reciever.
  L = torch.from_numpy(np.array([L]))
  L_SR = torch.cat((L,src,rcvr))
  L_SR.requires_grad = True
  
  ray = PINN.forward(L_SR)
  X_dx = autograd.grad(ray[0],L_SR,torch.ones_like(ray[0]),retain_graph=True)[0][0]
  X_dy = autograd.grad(ray[1],L_SR,torch.ones_like(ray[1]))[0][0]
  X_dL = (X_dx**2 + X_dy**2)**0.5 #Jacobian working now :)
  
  v = vel_model.forward(ray).detach().numpy()
  s = v**-1
  
  out = s*X_dL.detach().numpy()
  return out
  
def slowT(L,src,rcvr):
  #Function for travel time calculation using scipy.integrate.quad
  #Takes single L value, source and reciever. #Tensors already require grad!
  L = torch.from_numpy(np.array([L]))
  L_SR = torch.cat((L,src,rcvr))
  
  
  ray = PINN.forward(L_SR)
  X_dx = autograd.grad(ray[0],L_SR,torch.ones_like(ray[0]),retain_graph=True)[0][0]
  X_dy = autograd.grad(ray[1],L_SR,torch.ones_like(ray[1]))[0][0]
  X_dL = (X_dx**2 + X_dy**2)**0.5 #Jacobian working now :)
  
  v = vel_model.forward(ray).detach().numpy()
  s = v**-1
  
  out = s*X_dL.detach().numpy()
  return out

#Cacluate travel times for each ray
time_real = np.zeros(n_rays)
for i in range(n_rays):
  src_i = torch.from_numpy(src[i,:])
  rcvr_i = torch.from_numpy(rcvr[i,:])
  
  time_real[i] = integrate.quad(slow,0,1,(src_i,rcvr_i))[0]
  

##Define function that takes x,y coordinates and caluculates fit to data
def fit(xy):
  #print(xy)
  src_quake = np.zeros((n_rays,2))
  rcvr_quake = np.ones((n_rays,2))
  
  #Earthquake Source
  src_quake[:,0] = xy[0]
  src_quake[:,1] = xy[1]

  rcvr_quake[:,0] = np.linspace(0.05,0.95,n_rays) # Equally spaced recievers on the surface

  src_quakeT = torch.from_numpy(src_quake).to(device)
  rcvr_quakeT = torch.from_numpy(rcvr_quake).to(device)

  #Ray
  #Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y
  LambdaSR = np.zeros((n_rays*101,5))
  for i in range(n_rays):
    LambdaSR[i*101:101*(i+1),0] = np.linspace(0,1,101)
    LambdaSR[i*101:101*(i+1),1:3] = src_quake[i,:]
    LambdaSR[i*101:101*(i+1),3:5] = rcvr_quake[i,:]
    
    
  LambdaSR_T = torch.from_numpy(LambdaSR)
  LambdaSR_T = LambdaSR_T.to(device)
  
  ray = PINN.forward(LambdaSR_T)
    
  time_guess = np.zeros(n_rays)
  
  for i in range(n_rays):
    src_i = torch.from_numpy(src_quake[i,:])
    rcvr_i = torch.from_numpy(rcvr_quake[i,:])  
    time_guess[i] = integrate.quad(slow,0,1,(src_i,rcvr_i))[0]
  
  diff = np.mean((time_real-time_guess)**2)
  #print(diff)
  return diff
  
#Fit funciton using tensors
def fitT(x,y,time_real):
  src_quake = torch.zeros((n_rays,2))
  src_quake[:,0] = x
  src_quake[:,1] = y 
  
  rcvr_quake = torch.ones((n_rays,2))
  rcvr_quake[:,0] = torch.linspace(0.05,0.95,n_rays) # Equally spaced recievers on the surface
  
  src_quake = src_quake.to(device)
  rcvr_quake = rcvr_quake.to(device)
  n_lambda = 101
  
  #Ray
  #Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y
  LambdaSR = torch.zeros((n_rays*n_lambda,5))
  for i in range(n_rays):
    LambdaSR[i*n_lambda:n_lambda*(i+1),0] = torch.linspace(0,1,n_lambda)
    LambdaSR[i*n_lambda:n_lambda*(i+1),1:3] = src_quake[i,:]
    LambdaSR[i*n_lambda:n_lambda*(i+1),3:5] = rcvr_quake[i,:]
    
  LambdaSR = LambdaSR.to(device)  
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
    
  
  time_real = torch.from_numpy(time_real)
  diff = torch.mean((time_real-time_guess)**2)
  return diff



  
#Calculate average fit for each ray
#Calculate input tensor 
LambdaSR = np.zeros((n_rays*101,5))
for i in range(n_rays):
  LambdaSR[i*101:101*(i+1),0] = np.linspace(0,1,101)
  LambdaSR[i*101:101*(i+1),1:3] = src[i,:]
  LambdaSR[i*101:101*(i+1),3:5] = rcvr[i,:]

LambdaSR_T = torch.from_numpy(LambdaSR)
LambdaSR_T = LambdaSR_T.to(device)
    
loss_ray = np.zeros(n_rays)
for i in range(n_rays):
  loss_ray[i] = PINN.loss_PDE(LambdaSR_T[i*101:101*(i+1),:])#mean loss for each ray
  
  
#out = optim.minimize(fit,(0.8,0.02))

##Plot misfit & Jacobian & hessian surface
xUB = 1 #x upper bound
yUB = 1 #y lower bound

n_points = 51

x = np.linspace(0,xUB,n_points)
y = np.linspace(0,yUB,n_points)

x_plot = np.expand_dims(x,1)
y_plot = np.expand_dims(y,1)
# Create the mesh 
X,Y=np.meshgrid(x,y)

xy_stack = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
xy_fit = np.zeros_like(xy_stack[:,0])#
J = np.zeros((np.shape(xy_stack)[0],2)) #Jacobian J[:,0] = dz/dx, J[:,1] = dz/dy
H = np.zeros((np.shape(xy_stack)[0],4)) #Hessian H[:,0] = d2z/dx2, H[:,1] = d2z/dxdy, H[:,2] = d2z/dydx, H[:,3] = d2z/dy2

for i in range(np.shape(xy_stack)[0]):
  x = torch.tensor([xy_stack[i,0]])
  y = torch.tensor([xy_stack[i,1]])
  x.requires_grad = True
  y.requires_grad=True
  
  z = fitT(x,y,time_real)
  xy_fit[i] = z.detach().numpy()
  
  Jx = autograd.grad(z,x,torch.ones_like(z), retain_graph=True, create_graph=True)[0]
  Jy = autograd.grad(z,y,torch.ones_like(z), retain_graph=True, create_graph=True)[0]
  J[i,0] = Jx.detach().numpy()
  J[i,1] = Jy.detach().numpy()
  
  Hxx = autograd.grad(Jx,x,torch.ones_like(Jx), retain_graph=True, create_graph=True)[0]
  Hxy = autograd.grad(Jx,y,torch.ones_like(Jx), retain_graph=True, create_graph=True)[0]
  Hyx = autograd.grad(Jy,x,torch.ones_like(Jy), retain_graph=True, create_graph=True)[0]
  Hyy = autograd.grad(Jy,y,torch.ones_like(Jy), retain_graph=True, create_graph=True)[0]
  H[i,0] = Hxx.detach().numpy()
  H[i,1] = Hxy.detach().numpy()  
  H[i,2] = Hyx.detach().numpy()  
  H[i,3] = Hyy.detach().numpy()  
  
  if i%10 == 0:
    print(i)
    

np.savetxt('xy_fit_new.csv', xy_fit, delimiter=',')  
np.savetxt('Jacobian.csv', J, delimiter=',')
np.savetxt('Hessian.csv', H, delimiter=',')












