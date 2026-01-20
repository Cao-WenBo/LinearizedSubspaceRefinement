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
from util import diff2d

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

class TSONN():

    def __init__(self, layers, device, folder_name):
        
        self.Nx = 257
        self.Nt = 101
        self.layers = layers
        self.device = device
        self.folder_name = folder_name
        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        shutil.copyfile(__file__, self.folder_name + '/'+ os.path.basename(__file__))
        
        t = torch.linspace(0.0, 1.0, self.Nt, device=self.device)
        x = torch.linspace(-1.0, 1.0, self.Nx, device=self.device)
        xx,tt = torch.meshgrid(x,t)
        
        self.dx = x[1] - x[0]; self.dt = t[1] - t[0]; 
        self.X_ref = torch.cat([xx.reshape(-1,1),tt.reshape(-1,1)],dim=1).to(self.device); 
        
        self.X_ic = torch.cat([xx[:,[0]],tt[:,[0]]],dim=1).to(self.device)
        self.u_ic =  - torch.sin(torch.pi*xx[:,[0]])
        
        self.X_lbc = torch.cat([xx[[0]],tt[[0]]],dim=0).T.to(self.device)
        self.X_ubc = torch.cat([xx[[-1]],tt[[-1]]],dim=0).T.to(self.device)

        self.log = {'losses':[], 'losses_b':[], 'losses_i':[], 'losses_f':[], 'res':[], 'res_linear':[], 'error':[], 'error_linear':[], 'time':[]}

        self.log_result = []
        
        uu_ref = np.loadtxt('u_ref.dat').T
        self.u_ref = torch.tensor(uu_ref).to(device).reshape(-1,1).float()
        
        self.model = Net(self.layers, self.X_ref, self.device).to(self.device)

        self.dtau_inf = 1e10
        self.dtau = self.dtau_inf

    def forward(self, params_vector, X):
        params_dict = vector_to_param_dict(self.model, params_vector)
        f = functional_call(self.model, params_dict, X)
        return f
    
    def TimeStepping(self):
        X = self.X_pde

        f0, f_linear = jvp(lambda params: self.forward(params, X), (self.params_vector_base,), (self.params_vector_delta,))
        pred = f0 + f_linear

        self.U0_pde = pred.detach()
        
    def System_U(self, params_vector):
        X = self.X_batch
        def forward_U(X):
            U = self.forward(params_vector, X)
            return U
        
        U = vmap(forward_U)(X)
        U = U.squeeze(1)
        
        [U_ic, U_lbc, U_ubc, U_pde] = torch.split(U, self.split_sizes)
        
        UU = torch.cat([U_ic, U_ubc, U_lbc, U_pde])
        return UU

    def System(self, params_vector):
        X = self.X_batch
        def forward_U(X):
            U = self.forward(params_vector, X)
            return U, U
        
        def forward_DU(X):
            DU, U = jacrev(forward_U, has_aux=True)(X)
            DU_partial = DU[:,0,0:1]
            return DU_partial, (U, DU)
        
        def forward_DDU(X):
            DDU_partial, (U, DU) = jacrev(forward_DU, has_aux=True)(X)
            return U, DU, DDU_partial
        
        U, DU, DDU_partial = vmap(forward_DDU)(X)
        U = U.squeeze(1); DU = DU.squeeze(1); DDU_partial = DDU_partial.squeeze(1)
            
        [U_ic, U_lbc, U_ubc, U_pde] = torch.split(U, self.split_sizes)
        [DU_ic, DU_lbc, DU_ubc, DU_pde] = torch.split(DU, self.split_sizes)
        [_, _, _, DDU_partial_pde] = torch.split(DDU_partial, self.split_sizes)
        
        r_ic = U_ic - self.u_ic
        
        r_bc1 = U_ubc - U_lbc
        r_bc2 = DU_ubc[:,0:1,0] - DU_lbc[:,0:1,0]
        
        u = U_pde
        u_x = DU_pde[:,0:1,0]
        u_t = DU_pde[:,0:1,1]
        u_xx = DDU_partial_pde[:,0:1,0]

        r_pde = u_t + u*u_x - 0.01/torch.pi*u_xx

        r_pde += (u - self.U0_pde) / self.dtau

        w_ic = (r_pde.shape[0]/r_ic.shape[0])**0.5
        w_bc1 = (r_pde.shape[0]/r_bc1.shape[0])**0.5
        w_bc2 = (r_pde.shape[0]/r_bc2.shape[0])**0.5

        UU = torch.cat([U_ic, U_ubc, U_lbc, U_pde])
        RR = torch.cat([1*w_ic*r_ic, 1*w_bc1*r_bc1, 1*w_bc2*r_bc2, r_pde])

        return RR
    
    def Loss(self):
        f = self.System(self.params_vector_opt)
        loss = (f**2).mean()
        return loss
    
    def Preconditioning(self, rank=100, power_iter=1):
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()
        
        rank = 1000
        oversample = 10
        k = rank + oversample
        chunk_size = 32

        def AJV_fn(params_vector_batch, chunk_size=chunk_size):
            VV = params_vector_batch.T
            def Linearization(params_vector_delta):
                f0, f_linear = jvp(self.System, (self.params_vector_base,), (params_vector_delta,))
                return f_linear
            JV = torch.cat([vmap(Linearization)(params_vector_chunk)
                            for params_vector_chunk in VV.split(chunk_size)]).squeeze()
            return JV.T
        
        def AJTV_fn(gradient_batch, chunk_size=chunk_size):
            gradient_batch = gradient_batch.T
            y, vjp_fn = vjp(lambda params: self.System(params).flatten(), self.params_vector_base)
            JTV =  torch.cat([vmap(vjp_fn)(gradient_chunk)[0]
                            for gradient_chunk in gradient_batch.split(chunk_size)]).squeeze()
            return JTV.T

        def JV_fn(params_vector_batch, chunk_size=chunk_size):
            VV = params_vector_batch.T
            def Linearization(params_vector_delta):
                f0, f_linear = jvp(self.System_U, (self.params_vector_base,), (params_vector_delta,))
                return f_linear
            
            JV = torch.cat([vmap(Linearization)(params_vector_chunk)
                            for params_vector_chunk in VV.split(chunk_size)]).squeeze()
            return JV.T
        
        def JTV_fn(gradient_batch, chunk_size=chunk_size):
            gradient_batch = gradient_batch.T
            y, vjp_fn = vjp(lambda params: self.System_U(params).flatten(), self.params_vector_base)
            JTV =  torch.cat([vmap(vjp_fn)(gradient_chunk)[0]
                            for gradient_chunk in gradient_batch.split(chunk_size)]).squeeze()
            return JTV.T

        with torch.no_grad():
            f0 = self.System(self.params_vector_base)
            Omega = torch.rand(self.params_vector_base.shape[0], k, dtype=self.X_batch.dtype, device=self.device)  # 初始随机向量

            ## mode 1
            JO = JV_fn(Omega)
            Q, R = torch.linalg.qr(JO, mode='reduced')
            JTQ = JTV_fn(Q)

            U, S, Vh = torch.linalg.svd(JTQ.T, full_matrices=False)
            # V = (Vh.T@torch.diag(1 / S))[:,:rank]
            V = (Vh.T)[:,:rank]

            AJV = AJV_fn(V)
            # AJV_inv = torch.pinverse(AJV)
            # AJV_inv = torch.linalg.pinv(AJV, rtol=1e-15)
            # self.params_vector_delta = (V@(AJV_inv@(-f0))).flatten()
            self.params_vector_delta = (V@torch.linalg.lstsq(AJV, -f0, rcond=1e-15).solution).flatten()
            print(f"Cond: {torch.linalg.cond(AJV).item():.0f}", end=' ')  

            ## mode 2
            # JO = AJV_fn(Omega)
            # Q, R = torch.linalg.qr(JO, mode='reduced')
            # AJTQ = AJTV_fn(Q)

            # U, S, Vh = torch.linalg.svd(AJTQ.T, full_matrices=False)
            # # V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

            # AJV = AJV_fn(V)

            # self.params_vector_delta = (V@torch.linalg.lstsq(AJV, -f0, rcond=1e-15).solution).flatten()
            # print(f"Cond: {torch.linalg.cond(AJV).item():.0f}", end=' ')  

            ## mode 3
            # JO = AJV_fn(Omega)
            # Q, R = torch.linalg.qr(JO, mode='reduced')
            # Q = Q[:, :rank]
            # AJTQ = AJTV_fn(Q)
            # self.params_vector_delta = (torch.linalg.lstsq(AJTQ.T, Q.T@(-f0), rcond=1e-15).solution).flatten()
            # print(f"Cond: {torch.linalg.cond(AJTQ).item():.0f}", end=' ') 

            ## mode 4
            # JO = AJV_fn(Omega)
            # Q, R = torch.linalg.qr(JO, mode='reduced')
            # AJTQ = AJTV_fn(Q)

            # U, S, Vh = torch.linalg.svd(AJTQ.T, full_matrices=False)
            # # V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

            # AJTQ = AJTQ[:, :rank]; Q = Q[:, :rank]
 
            # self.params_vector_delta = (V@torch.linalg.lstsq(AJTQ.T@V, Q.T@(-f0), rcond=1e-15).solution).flatten()
            # print(f"Cond: {torch.linalg.cond(AJTQ).item():.0f}", end=' ') 

        end_mem = torch.cuda.memory_allocated()
        end_peak = torch.cuda.max_memory_allocated()

        print(f"Memory: {end_peak / 1024**3:.2f} GB", end=' ')

    def closure(self):
        if self.params_vector_opt.grad is not None:
            self.params_vector_opt.grad.detach_()
            self.params_vector_opt.grad.zero_()
        self.loss = self.Loss()
        self.loss.backward()
        return self.loss

    def CollectionPoints(self):
        # PDE
        X = torch.rand(20000,2, device=self.device)
        X[:,0:1] =  X[:,0:1] * 2 - 1
        self.X_pde = X
        # IC
        x = torch.rand(1000,1, device=self.device) * 2 - 1
        self.X_ic = torch.cat([x, x*0], dim=1)
        self.u_ic =  - torch.sin(torch.pi*self.X_ic[:,[0]])
        # BC
        t = torch.rand(1000,1, device=self.device)
        self.X_lbc = torch.cat([t*0-1, t], dim=1)
        self.X_ubc = torch.cat([t*0+1, t], dim=1)

        self.X_batch = torch.cat([self.X_ic, self.X_lbc, self.X_ubc, self.X_pde])
        self.split_sizes = [self.X_ic.shape[0], self.X_lbc.shape[0], self.X_ubc.shape[0], self.X_pde.shape[0]]

    def LogRecord(self):

        loss = self.Loss()

        self.dtau = self.dtau_inf
        pred = self.forward(self.params_vector_opt, self.X_ref)
        self.log_result.append(pred.cpu().detach().numpy())
        error = torch.norm(pred - self.u_ref) / torch.norm(self.u_ref)
        res = (self.System(self.params_vector_opt)**2).mean()

        f0, f_linear = jvp(lambda params: self.forward(params, self.X_ref), (self.params_vector_base,), (self.params_vector_delta,))
        pred = f0 + f_linear
        error_linear = torch.norm(pred - self.u_ref) / torch.norm(self.u_ref)

        self.log_result.append(pred.cpu().detach().numpy())

        f0, f_linear = jvp(self.System, (self.params_vector_base,), (self.params_vector_delta,))
        res_linear = ((f0 + f_linear)**2).mean()

        t2 = time.time()
        self.log['error'].append(error.item())
        self.log['error_linear'].append(error_linear.item())
        self.log['res'].append(res.item())
        self.log['res_linear'].append(res_linear.item())
        self.log['losses'].append(loss.item())
        self.log['time'].append(t2-self.t1)

    def train(self, epoch):

        if len(self.log['time']) == 0:
            self.t1 = time.time()
        else:
            self.t1 = time.time() - self.log['time'][-1]

        self.params_vector_base = parameters_to_vector(self.model.parameters()).detach().clone()
        self.params_vector_delta = torch.zeros_like(self.params_vector_base)
        self.params_vector_opt = self.params_vector_base * 1; self.params_vector_opt.requires_grad = True
        self.CollectionPoints()
        self.TimeStepping()
        self.LogRecord()
        
        for i in range(epoch):
            self.niter = i
            def closure():
                self.optimizer.zero_grad()
                self.loss = self.Loss()
                self.loss.backward()
                return self.loss
            
            self.params_vector_base = parameters_to_vector(self.model.parameters()).detach().clone()
            self.params_vector_delta = torch.zeros_like(self.params_vector_base)
            self.params_vector_opt = self.params_vector_base * 1; self.params_vector_opt.requires_grad = True

            self.CollectionPoints()
            self.dtau = self.dtau_inf
            self.TimeStepping()

            self.Preconditioning(rank=1000)


            self.TimeStepping()

            # 非线性模型
            self.use_linear = False
            self.dtau = 0.3
            self.optimizer = torch.optim.LBFGS([self.params_vector_opt], max_iter=500, history_size=500,
                                                tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe') #
            self.optimizer.step(self.closure)
            
            self.LogRecord()

            vector_to_parameters(self.params_vector_opt, self.model.parameters())
        
            print(f'{i}|{epoch} loss={self.log["losses"][-1]:.4g} error_linear={self.log["error_linear"][-1]:.4g} res_linear={self.log["res_linear"][-1]:.4g}', end=' ')
            print(f'error={self.log["error"][-1]:.4g} res={self.log["res"][-1]:.4g} time={self.log["time"][-1]:.2f}')

        save_class(self, self.folder_name+'/10000.pkl')

if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if 1 else "cpu")
    

    layers = [2] + [128] * 6 + [1]

    nn = TSONN(layers, device, folder_name='A117-v4-20000-1000-dtau0.3-R1000-opt500-500-mode1-pre-saveresult')

    nn.train(50)
    
    t2 = time.time()
    print(t2 - t1)