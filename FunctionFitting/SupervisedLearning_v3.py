import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import vmap, jacrev, hessian, jvp, vjp, functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from util import fwd_gradients, load_class, save_class, load_class_name
import time
import pickle
import os 
import shutil
import time
import copy

# Solving the Linearied System using Optimization

def vector_to_param_dict(model, theta_vec):
    param_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param_dict[name] = theta_vec[pointer:pointer + numel].view_as(param)
        pointer += numel
    return param_dict

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class Net(torch.nn.Module):
    def __init__(self, layers, X, device):
        super(Net, self).__init__()
        
        self.X_mean = X.mean(0, keepdim=True)
        self.X_std = X.std(0, keepdim=True)
    
        self.num_layers = len(layers)
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers.to(device)
        
    def forward(self, x):
        x = ((x - self.X_mean) / self.X_std) # z-score norm
        for i in range(0, self.num_layers-1):
            x = self.layers[i](x)
            if i < self.num_layers-2:
                x = torch.tanh(x)
        return x

class SupervisedLearning():

    def __init__(self, layers, device, folder_name):
        self.layers = layers
        self.device = device
        self.folder_name = folder_name
        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        shutil.copyfile(__file__, self.folder_name + '/'+ os.path.basename(__file__))

        self.k = 1
        
        self.X_data = torch.rand(10000,2, device=self.device) * 2 - 1
        self.Y_data = self.fn_tar(self.X_data)

        self.log = {'train_loss':[], 'val_loss':[], 'test_loss':[], 
                    'train_loss_LSR':[], 'val_loss_LSR':[], 'test_loss_LSR':[], 'time':[]}
        
        self.model = Net(self.layers, self.X_data, self.device).to(self.device)
        
        self.Nx = 100
        self.Ny = 100
        x = torch.linspace(-1, 1, self.Nx, device=self.device)
        y = torch.linspace(-1, 1, self.Nx, device=self.device)
        xx,yy = torch.meshgrid(x,y)
        
        self.X_ref = torch.cat([xx.reshape(-1,1),yy.reshape(-1,1)],dim=1).to(self.device)
        self.Y_ref = self.fn_tar(self.X_ref)
        
    def fn_tar(self, X):
        x = X[:,0:1]; y = X[:,1:2]
        return torch.sin(self.k*torch.pi*x) * torch.sin(self.k*torch.pi*y)

    def forward(self, params_vector, X):
        params_dict = vector_to_param_dict(self.model, params_vector)
        f = functional_call(self.model, params_dict, X)
        return f

    def System(self, params_vector):
        X = self.X_batch
        def forward_U(X):
            U = self.forward(params_vector, X)
            return U
        
        U = vmap(forward_U)(X)
        U = U.squeeze(1)
        
        R = U - self.Y_batch
        return R
    
    def LSR(self, rank=1000):
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()
        
        oversample = 10
        k = rank + oversample
        chunk_size = 600  # This parameter depends on the size of the GPU memory.

        def JV_fn(params_vector_batch, chunk_size=chunk_size):
            VV = params_vector_batch.T
            def Linearization(params_vector_delta):
                f0, f_linear = jvp(self.System, (self.params_vector_base,), (params_vector_delta,))
                return f_linear
            
            JV = torch.cat([vmap(Linearization)(params_vector_chunk)
                            for params_vector_chunk in VV.split(chunk_size)]).squeeze()
            return JV.T
        
        def JTV_fn(gradient_batch, chunk_size=chunk_size):
            gradient_batch = gradient_batch.T
            y, vjp_fn = vjp(lambda params: self.System(params).flatten(), self.params_vector_base)
            JTV =  torch.cat([vmap(vjp_fn)(gradient_chunk)[0]
                            for gradient_chunk in gradient_batch.split(chunk_size)]).squeeze()
            return JTV.T

        with torch.no_grad():
            self.X_batch = self.X_train; self.Y_batch = self.Y_train
            f0 = self.System(self.params_vector_base)
            Omega = torch.rand(self.params_vector_base.shape[0], k, dtype=self.X_data.dtype, device=self.device)  # 初始随机向量

            ## mode 1
            JO = JV_fn(Omega)
            Q, R = torch.linalg.qr(JO, mode='reduced')
            JTQ = JTV_fn(Q)

            U, S, Vh = torch.linalg.svd(JTQ.T, full_matrices=False)
            V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

            JV = JV_fn(V)

            self.params_vector_delta = (V@torch.linalg.lstsq(JV, -f0, rcond=1e-15).solution).flatten()
            print(f"Cond: {torch.linalg.cond(JV).item():.0f}", end=' ')  

            ## mode 3
            # JO = JV_fn(Omega)
            # Q, R = torch.linalg.qr(JO, mode='reduced')
            # Q = Q[:, :rank]
            # JTQ = JTV_fn(Q)
            # self.params_vector_delta = (torch.linalg.lstsq(JTQ.T, Q.T@(-f0), rcond=1e-15).solution).flatten()
            # print(f"Cond: {torch.linalg.cond(JTQ).item():.0f}", end=' ') 


        end_mem = torch.cuda.memory_allocated()
        end_peak = torch.cuda.max_memory_allocated()

        print(f"Memory: {end_peak / 1024**3:.2f} GB", end=' ')

    def train(self, epochs=500, lr=1e-4):

        self.params_vector_delta *= 0
        self.params_vector_delta.requires_grad = True
        
        # optimizer = torch.optim.Adam([self.params_vector_delta], lr=lr)
        optimizer = torch.optim.LBFGS([self.params_vector_delta], max_iter=100, history_size=1000, tolerance_grad=1e-10, tolerance_change=1e-12) # lr=1
        criterion = torch.nn.MSELoss()

        t0 = time.time()
        for epoch in range(1, epochs+1):
            def Pred(X):
                f0, f_linear = jvp(lambda params: self.forward(params, X), (self.params_vector_base,), (self.params_vector_delta,))
                return f0 + f_linear

            def closure():
                optimizer.zero_grad()
                pred = Pred(self.X_train)
                train_loss = criterion(pred, self.Y_train) * 1e6
                train_loss.backward()
                return train_loss
            
            # Full batch opt
            train_loss = optimizer.step(closure) / 1e6 # LBFGS
            # for k in range(100): train_loss = optimizer.step(closure)

            
            with torch.no_grad():
                val_pred = Pred(self.X_val)
                val_loss = criterion(val_pred, self.Y_val).item()

            # scheduler.step(val_loss)

            with torch.no_grad():
                test_pred = Pred(self.X_ref)
                test_loss = criterion(test_pred, self.Y_ref).item()

            # Logging
            self.log['train_loss'].append(train_loss)
            self.log['val_loss'].append(val_loss)
            self.log['test_loss'].append(test_loss)
            self.log['time'].append(time.time() - t0)
        
            # if epoch % 100 == 0: 
            print(f'Epoch {epoch}/{epochs}: train_loss={train_loss:.4g}, val_loss={val_loss:.4g}, test_loss={test_loss:.4g}', end=' ')
            print(f'time={self.log["time"][-1]:.2f}')

        save_class(self, self.folder_name+'/10000.pkl')

if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if 1 else "cpu")
    
    layers = [2, 128, 128, 128, 128, 128, 128, 1]

    nn = load_class('A200-v2-k1-mode1-pre/10000.pkl') # load a trained obj
    nn.folder_name = 'A221-Linearied-Opt-LBFGS-history-1000'
    if not os.path.exists(nn.folder_name):
        os.makedirs(nn.folder_name)
    shutil.copyfile(__file__, nn.folder_name + '/'+ os.path.basename(__file__))
    nn.log = {'train_loss':[], 'val_loss':[], 'test_loss':[], 'time':[]}
    nn.train(10000, lr=1e-7)

    t2 = time.time()
    print(t2 - t1)


