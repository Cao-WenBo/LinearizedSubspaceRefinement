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

# Solving the Linearied System using Iteration

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

    def LogRecord(self, k, params_vector_delta):
        def Pred(X):
            f0, f_linear = jvp(lambda params: self.forward(params, X), (self.params_vector_base,), (self.params_vector_delta,))
            return f0 + f_linear

        self.params_vector_delta = params_vector_delta
        criterion = torch.nn.MSELoss()
        train_pred = Pred(self.X_train)
        train_loss = criterion(train_pred, self.Y_train).item()

        val_pred = Pred(self.X_val)
        val_loss = criterion(val_pred, self.Y_val).item()

        test_pred = Pred(self.X_ref)
        test_loss = criterion(test_pred, self.Y_ref).item()

        self.log['train_loss'].append(train_loss)
        self.log['val_loss'].append(val_loss)
        self.log['test_loss'].append(test_loss)
        self.log['time'].append(time.time() - self.t0)

        print(f'Epoch {k}: train_loss={train_loss:.4g}, val_loss={val_loss:.4g}, test_loss={test_loss:.4g}', end=' ')
        print(f'time={self.log["time"][-1]:.2f}')

    def train(self, epochs=100):
        self.t0 = time.time()
        self.X_batch = self.X_train; self.Y_batch = self.Y_train
        
        def Jv_fn(params_vector_delta):
            f0, f_linear = jvp(self.System, (self.params_vector_base,), (params_vector_delta,))
            return f_linear.flatten()
            
        def JTv_fn(gradient):
            y, vjp_fn = vjp(lambda params: self.System(params).flatten(), self.params_vector_base)
            return vjp_fn(gradient.flatten())[0]
        
        def CGLS(Jv, JTz, r, max_iter=5000, tol=1e-10):
            d = torch.zeros_like(JTz(r))
            z = JTz(r)                   # z = J^T r
            p = z.clone()
            gamma = torch.dot(z, z)
        
            for k in range(max_iter):
                Jp = Jv(p)
                alpha = gamma / torch.dot(Jp, Jp)
        
                d = d + alpha * p
                r = r - alpha * Jp
                z = JTz(r)
        
                gamma_new = torch.dot(z, z)
                if torch.sqrt(gamma_new) < tol:
                    break
        
                beta = gamma_new / gamma
                p = z + beta * p
                gamma = gamma_new

                if k % 100 == 0: self.LogRecord(k, d)
        
            return d
        
        def CGNR(Jv, JTz, r, max_iter=5000, tol=1e-10):
            # Solve min ||J d - r|| by CG on J J^T residuals
            u = r.clone()                    # initial residual r
            v = JTz(u)                       # v = J^T u
            d = torch.zeros_like(v)
            p = v.clone()
        
            gamma = torch.dot(u, u)
        
            for k in range(max_iter):
                Jp = Jv(p)
                alpha = gamma / torch.dot(Jp, Jp)
        
                d = d + alpha * p
                u = u - alpha * Jp           # update residual
                v = JTz(u)
        
                gamma_new = torch.dot(u, u)
                if torch.sqrt(gamma_new) < tol:
                    break
        
                beta = gamma_new / gamma
                p = v + beta * p
                gamma = gamma_new

                if k % 100 == 0: self.LogRecord(k, d)
            return d

        def LSQR(Jv, JTz, r, max_iter=5000, tol=1e-10):
            # Initialization
            u = r.clone()
            beta = torch.norm(u)
            u = u / beta
        
            v = JTz(u)
            alpha = torch.norm(v)
            v = v / alpha
        
            w = v.clone()
            d = torch.zeros_like(v)
        
            phibar = beta
            rhobar = alpha
        
            for k in range(max_iter):
                # bidiagonalization
                u = Jv(v) - alpha * u
                beta = torch.norm(u)
                if beta > 0:
                    u = u / beta
        
                v = JTz(u) - beta * v
                alpha = torch.norm(v)
                if alpha > 0:
                    v = v / alpha
        
                # QR step
                rho = torch.sqrt(rhobar * rhobar + beta * beta)
                c = rhobar / rho
                s = beta / rho
                theta = s * alpha
                rhobar = -c * alpha
                phi = c * phibar
                phibar = s * phibar
        
                d = d + (phi / rho) * w
                w = v - (theta / rho) * w

                if k % 100 == 0: self.LogRecord(k, d)
        
                if torch.abs(phibar) < tol:
                    break
        
            return d
                
        def LSMR(Jv, JTz, r, max_iter=5000, tol=1e-10):
            u = r.clone()
            beta = torch.norm(u)
            u = u / beta
        
            v = JTz(u)
            alpha = torch.norm(v)
            v = v / alpha
        
            d = torch.zeros_like(v)
            h = v.clone()
            hbar = torch.zeros_like(v)
        
            rhobar = alpha
            betabar = beta
            phi = beta
            rhobar_old = 0.0
            c_old = 1.0
            s_old = 0.0
        
            for k in range(max_iter):
                u = Jv(v) - alpha * u
                beta = torch.norm(u)
                if beta != 0:
                    u = u / beta
        
                v = JTz(u) - beta * v
                alpha = torch.norm(v)
                if alpha != 0:
                    v = v / alpha
        
                # rotation
                rho = torch.sqrt(rhobar * rhobar + beta * beta)
                c = rhobar / rho
                s = beta / rho
                theta = s * alpha
                rhobar = -c * alpha
                phi_old = phi
                phi = c * phi
        
                # update
                d = d + (phi_old / rho) * h
                hbar = v - (theta / rho) * h
                h = hbar.clone()

                if k % 100 == 0: self.LogRecord(k, d)
        
                if torch.abs(phi) < tol:
                    break
        
            return d

        def MINRES(Jv, JTz, r, max_iter=5000, tol=1e-10):
            # x = [u; d]
            x_u = torch.zeros_like(r)
            x_d = torch.zeros_like(JTz(r))
        
            # initial residual
            r_u = r.clone()
            r_d = torch.zeros_like(x_d)
        
            # MINRES vectors
            v_u = r_u.clone()
            v_d = r_d.clone()
        
            beta = torch.sqrt(torch.dot(v_u, v_u) + torch.dot(v_d, v_d))
        
            v_u = v_u / beta
            v_d = v_d / beta
        
            # previous vectors
            v_u_old = torch.zeros_like(v_u)
            v_d_old = torch.zeros_like(v_d)
        
            # recursion
            for k in range(max_iter):
                # A*[v_u; v_d]
                Av_u = Jv(v_d)
                Av_d = JTz(v_u)
        
                # MINRES Lanczos
                alpha = torch.dot(v_u, Av_u) + torch.dot(v_d, Av_d)
        
                w_u = Av_u - alpha * v_u - beta * v_u_old
                w_d = Av_d - alpha * v_d - beta * v_d_old
        
                beta_new = torch.sqrt(torch.dot(w_u, w_u) + torch.dot(w_d, w_d))
        
                # update
                v_u_old = v_u
                v_d_old = v_d
        
                v_u = w_u / beta_new
                v_d = w_d / beta_new
        
                # MINRES update x
                x_u = x_u + alpha * v_u_old
                x_d = x_d + alpha * v_d_old

                if k % 100 == 0: self.LogRecord(k, x_d)
        
                if beta_new < tol:
                    break
        
                beta = beta_new
        
            return x_d   # return d

        
        with torch.no_grad():
            f0 = self.System(self.params_vector_base).flatten()

            self.params_vector_delta = (CGLS(Jv_fn, JTv_fn, -f0, max_iter=int(epochs), tol=1e-16)).flatten() 

        save_class(self, self.folder_name+'/10000.pkl')

if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if 1 else "cpu")
    
    layers = [2, 128, 128, 128, 128, 128, 128, 1]

    nn = load_class('A200-v2-k1-mode1-pre/10000.pkl') # load a trained obj
    nn.folder_name = 'A214-Linearied-Iteration-MINRES'
    if not os.path.exists(nn.folder_name):
        os.makedirs(nn.folder_name)
    shutil.copyfile(__file__, nn.folder_name + '/'+ os.path.basename(__file__))
    nn.log = {'train_loss':[], 'val_loss':[], 'test_loss':[], 'time':[]}
    nn.train(1e6)

    t2 = time.time()
    print(t2 - t1)


