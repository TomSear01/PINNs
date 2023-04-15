# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:20:27 2022

@author: thomas

Plots appearence:
1. 'Real' Data (from forward model)
2. Sampled Data training points
3. Sampled Physics Training points
4. PINN before training
5. PINN after training
6. Difference between PINN after training and real data


"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d                  
import time
import numpy as np

#---------------------------SET UP-------------------------
torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80701)
np.random.seed(80701)

#Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    
#-----------------------DEFINE FUNCTIONS-----------------
#Forward modelling function
def FEM_wave(x,tme,v):
 
    dx = x[2]-x[1]      #Grid spacing (assume fixed)
    

    #Time discretisation
    dt = dx/np.max(v)   #Timestep in sec (critical timestep?)

    #%Initial Conditions
    U = 1               #Wave amplitude

    #Gaussian Pulse
    x0 = length/2
    w = 0.03*length
    #uold = U*np.exp(-((x-x0)/w)**2)
    
    #Standing Wave
    uold = U*np.sin(2*np.pi*x/length)

    uolder = uold




    def waveprop2(uold, uolder, dx, dt, v):
        """performs 1 eave eq timestep
        """
        #Calculate double derivative of u to x:
        d2fdx2=np.diff(uold,2,0)/dx**2
        #Add boundary points
        d2fdx2 = np.insert(d2fdx2,0,0)
        d2fdx2 = np.append(d2fdx2,0)

        unew = 2*uold - uolder +dt**2*v**2*d2fdx2
        return unew
    
    #Create storage matrix
    store = np.zeros((nt, nx))
    store[0, :] = uold


    for it in range(1, nt):
        #Two numerical Solutions!
        #Simple double derivative
            
        #Using diff function
        unew = waveprop2(uold, uolder, dx, dt, v)
        #Store Values
        store[it, :] = unew
        uolder = uold
        uold = unew

    store1 = np.swapaxes(store,0,1)

    return store1
#Various Plotting Functions. Combine?
def plot3D(x,t,y):
    x_plot =x.squeeze(1) 
    t_plot =t.squeeze(1)
    X,T= torch.meshgrid(x_plot,t_plot)
    F_xt = y
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

def plotTraining(x,t,y,x_train,t_train):
    x_plot =x.squeeze(1) 
    t_plot =t.squeeze(1)
    X,T= torch.meshgrid(x_plot,t_plot)
    F_xt = y
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
    plt.scatter(t_train,x_train,marker='.', c='y', linewidths=0.5)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    plt.savefig('outputs/sampled_data.png')
    
def plot3D_Matrix(x,t,y):
  X,T= x,t
  F_xt = y
  fig,ax=plt.subplots(1,1)
  cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title('F(x,t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  plt.show()
  plt.savefig('outputs/final_plot.png')

#------------------GENERATE DATA------------------------------------

length = 5000       #Length of model (string)
nx = 101            #Nr of grid points
x = np.linspace(0, length, nx) #Array for the finite difference mesh
dx = x[1]-x[0]

v = 1000           #Seismic velocity (m/s)

tmax = 5

dt = dx/np.max(v)   #Timestep in sec (critical timestep)
tmax = 5
nt = tmax/dt
nt = nt.astype('int')

t = np.linspace(0, tmax, nt) #Create time array


y_solution = FEM_wave(x,t,v) #Get data from forward model


x = np.expand_dims(x,1)
t = np.expand_dims(t,1)

# Create the mesh 
X,T=np.meshgrid(x,t)

#plot 'real' data
#plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(y_solution))

#--------------PREPARE DATA-----------------
#Transform the mesh into a 2 column vector [x,t]
xt_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) 
y_true = y_solution.flatten('F')[:,None]

#Domain Bounds
lb = xt_test[0]
ub = xt_test[-1]

#Get Training Data
##Random points
num_samples = 1000
inx = np.random.choice(xt_test.shape[0],num_samples, replace=False) # Get index of sample points
inx = np.sort(inx)
xt_train = xt_test[inx,:] #Get training data

##Fixed X points
#num_loc = 5
#loc = np.random.choice(nx,num_loc, replace=False)
#loc = x[loc]
#a = xt_test[:,0]
#inx = np.nonzero(np.in1d(a,loc))[0]
#xt_train = xt_test[inx,:]

#plotTraining(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(y_solution),xt_train[:,0],xt_train[:,1])
y_train = y_true[inx]

##Physics Data
#Physics Training Points
num_physics = np.size(X)
inx2 = np.random.choice(xt_test.shape[0],num_physics, replace=False)
inx2 = np.sort(inx2)
xt_physics = xt_test[inx2,:]

#plotTraining(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(y_solution),xt_physics[:,0],xt_physics[:,1])

#-----------------NEURAL NETWORK---------------------
class PINN(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Tanh() #Softplus
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        #Initialise neural network as a list using nn.Modulelist  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
   
          
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=3.0)
            #nn.init.xavier_uniform_(self.linears[i].weight.data, gain=1.0)
            #nn.init.normal_(self.linears[i].weight.data, mean=0, std=1)
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
        
        for i in range(len(layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)   
        return a
                        
    def loss_data(self,x,y):              
        loss_u = self.loss_function(self.forward(x), y)              
        return loss_u
  
    
    def loss_PDE(self, xt_physics):
                            
        xt = xt_physics.clone()                  
        xt.requires_grad = True    
        y = self.forward(xt) #Pass inputs through PINN
                   
        y_x_t = autograd.grad(y,xt,torch.ones([xt.shape[0], 1]).to(device), create_graph=True)[0]  #First Derivative  
        
        y_x = y_x_t[:,0]
        y_t = y_x_t[:,1]
        
        y_xx = autograd.grad(y_x, xt, torch.ones_like(y_x).to(device), create_graph=True)[0]
        y_xx = y_xx[:,0]
        
        y_tt = autograd.grad(y_t, xt, torch.ones_like(y_t).to(device), create_graph=True)[0]
        y_tt = y_tt[:,1]
        
                          

        eqn = ((v**2)*y_xx) - y_tt       
        loss_f = (eqn**2).mean()                
        return loss_f
    
    def loss(self,x,y,physics):

        loss_dat = self.loss_data(x,y)
        loss_phys = self.loss_PDE(physics)
        
        loss_val = alpha*loss_dat + (1-alpha)*loss_phys
        
        return loss_val, loss_dat, loss_phys
           
    
    #test neural network
    def test(self):
                
        y_pred = self.forward(xt_test)       
        error_vec = torch.linalg.norm((y-y_pred),2)/torch.linalg.norm(y,2)        # Relative L2 Norm of the error (Vector) (Not sure what this does, need to check!)  
        y_pred = y_pred.cpu().detach().numpy()   
        y_pred = np.reshape(y_pred,(nx,nt),order='F')            
        return error_vec, y_pred


#----------------TRAIN PINN-------------------
xt_train = torch.from_numpy(xt_train).float().to(device)
xt_physics = torch.from_numpy(xt_physics).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
xt_test = torch.from_numpy(xt_test).float().to(device)
y = torch.from_numpy(y_true).float().to(device)


lr=1e-3
layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers
alpha = 0.3

total_loss = []
epoch_count = []
phys_loss = []
data_loss = []

model = PINN(layers)       
model.to(device)


#Plot Initial Conditions
ybm = model(xt_test)
x1=xt_test[:,0]
t1=xt_test[:,1]

arr_x1=x1.reshape(shape=[nt,nx]).transpose(1,0).detach().cpu()
arr_T1=t1.reshape(shape=[nt,nx]).transpose(1,0).detach().cpu()
arr_ybm=ybm.reshape(shape=[nt,nx]).transpose(1,0).detach().cpu()
#plot3D_Matrix(arr_x1,arr_T1,arr_ybm)


optimiser = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)

start_time = time.time()

for i in range(5): # reducing learning rate after each
    print(f"\nLoop: {i+1}, lr:{lr}, alpha: {alpha}")
    for epoch in range(20000):
        optimiser.zero_grad()
        
        loss, loss_data, loss_physics = model.loss(xt_train, y_train, xt_physics)  
       
        
        if epoch%1000==0:
            epoch_count.append(epoch)
            total_loss.append(loss.detach().to('cpu').numpy())
            phys_loss.append(loss_physics.detach().to('cpu').numpy())
            data_loss.append(loss_data.detach().to('cpu').numpy())
            print(f"Epoch: {epoch} | Physics Loss: {(1-alpha)*loss_physics} | Data Loss: {alpha*loss_data} | total loss: {loss}")
       
        loss.backward()
        optimiser.step()
    # Drop learning rate by half
     
    
    for grp in optimiser.param_groups:
      grp['lr']*=0.5
      lr = grp['lr']
          
   
    


    
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))

#Model Accuracy  
error_vec, y_pred = model.test()

print('Test Error: %.5f'  % (error_vec))

x1=xt_test[:,0]
t1=xt_test[:,1]
     
arr_x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_T1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
arr_y1=y_pred


plot3D_Matrix(arr_x1,arr_T1,torch.from_numpy(arr_y1))

#Difference between model and real data
a = y_pred-y_solution

plot3D(torch.from_numpy(x),torch.from_numpy(t),a)


