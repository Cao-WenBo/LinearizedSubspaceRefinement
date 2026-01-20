import numpy as np
import torch
from torch.func import vmap, jacrev, hessian, jvp, vjp, functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
# from util import fwd_gradients, load_class, save_class, load_class_name
import time
import pickle
import os 
import shutil
import time
import copy
import scipy
from scipy.integrate import odeint
from JaxModel2TorchModel import TorchMLP, TorchPI_DeepONet

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def vector_to_param_dict(model, theta_vec):
    param_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param_dict[name] = theta_vec[pointer:pointer + numel].view_as(param)
        pointer += numel
    return param_dict

class LSR():
    def __init__(self, model, device, folder_name):
        self.model = model.to(device)
        self.device = device
        self.folder_name = folder_name
        
        # if not os.path.exists(self.folder_name):
        #     os.makedirs(self.folder_name)
        # shutil.copyfile(__file__, self.folder_name + '/'+ os.path.basename(__file__))
        
    def forward(self, params_vector, u, y):
        params_dict = vector_to_param_dict(self.model, params_vector)
        f = functional_call(self.model, params_dict, (u, y))
        return f
        
    def System_U(self, params_vector):
        UU = self.forward(params_vector, self.u_batch, self.y_batch)
        return UU

    def System(self, params_vector):
        # y = [x, t]
        # some symbol confuse: sometimes U is s, u is s, not always
        def forward_U(u, y):
            U = self.forward(params_vector, u, y)
            return U, U
        
        def forward_DU(u, y):
            DU, U = jacrev(forward_U, argnums=1, has_aux=True)(u, y)
            return U, DU
        
        U, DU = vmap(forward_DU)(self.u_batch, self.y_batch)
            
        [U_bc, U_pde] = torch.split(U, self.split_sizes)
        [DU_bc, DU_pde] = torch.split(DU, self.split_sizes)

        [s_ic, s_pde] = torch.split(self.s_batch, self.split_sizes)
        
        r_bc = U_bc - s_ic
    
        s_x = DU_pde[:,0:1,0]

        r_pde = s_x - s_pde
        
        N_bc = r_bc.shape[0]
        N_pde = r_pde.shape[0]
        N_tot = N_bc + N_pde
    
        # exact weights
        w_bc = (1 * N_tot / N_bc)**0.5
        w_pde = (1 * N_tot / N_pde)**0.5
    
        RR = torch.cat([w_bc*r_bc, w_pde*r_pde])
        return RR
    
    def Loss(self):
        f = self.System(self.params_vector_opt)
        loss = (f**2).mean()
        return loss
    
    def LSR(self, rank=100):
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()
        
        oversample = 10
        k = rank + oversample
        chunk_size = 1500 

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
            Omega = torch.rand(self.params_vector_base.shape[0], k, dtype=self.u_batch.dtype, device=self.device)  # 初始随机向量

            ## mode 1
            JO = JV_fn(Omega)
            Q, R = torch.linalg.qr(JO, mode='reduced')
            JTQ = JTV_fn(Q)

            U, S, Vh = torch.linalg.svd(JTQ.T, full_matrices=False)
            V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

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
        
    def compute_error_LSR(self, N_test, rank_LSR=50):
        
        self.log = {'error':[],'error_LSR':[],'res':[],'res_LSR':[],'time':[]}
        
        for i in range(N_test):

            gp_sample, u_test, y_test, s_test = generate_one_test_data_torch(P_test)
            
            u_train, y_train, s_train, u_r_train, y_r_train, s_r_train =  \
                generate_one_training_data_torch(gp_sample, P=P_train)
            
            u_train, y_train, s_train, u_r_train, y_r_train, s_r_train =  \
                torch.tensor(u_train, device=self.device).float(), torch.tensor(y_train, device=self.device).float(), \
                torch.tensor(s_train, device=self.device).float(), torch.tensor(u_r_train, device=self.device).float(), \
                torch.tensor(y_r_train, device=self.device).float(), torch.tensor(s_r_train, device=self.device).float()
                
            u_test, y_test, s_test = torch.tensor(u_test, device=self.device).float(), \
                torch.tensor(y_test, device=self.device).float(), torch.tensor(s_test, device=self.device).float()
            
            self.u_batch = torch.cat([u_train, u_r_train])
            self.y_batch = torch.cat([y_train, y_r_train])
            self.s_batch = torch.cat([s_train, s_r_train])
            
            
            self.split_sizes = [u_train.shape[0], u_r_train.shape[0]]
            
            
            self.params_vector_base = parameters_to_vector(self.model.parameters()).detach().clone()
            
            t1 = time.perf_counter()
            self.LSR(rank=rank_LSR)
            t2 = time.perf_counter()
            
            
    
            f0, f_linear = jvp(lambda params: self.forward(params, u_test, y_test), (self.params_vector_base,), (self.params_vector_delta,))
            pred = f0
            pred_linear = f0 + f_linear
            error = torch.linalg.norm(pred - s_test) / torch.linalg.norm(s_test)
            error_LSR = torch.linalg.norm(pred_linear - s_test) / torch.linalg.norm(s_test)
    

            f0, f_linear = jvp(lambda params: self.System(params), (self.params_vector_base,), (self.params_vector_delta,))
            res = (f0**2).mean()
            res_LSR = ((f0 + f_linear)**2).mean()

    
            self.log['error'].append(error.item())
            self.log['error_LSR'].append(error_LSR.item())
            self.log['res'].append(res.item())
            self.log['res_LSR'].append(res_LSR.item())
            self.log['time'].append(t2-t1)
            
            print(f'{i}|{N_test} error={error.item()} error_LSR={error_LSR.item()} res={res.item()} res_LSR={res_LSR.item()} time={t2-t1}')


def RBF_np(x1, x2, params):
    output_scale, lengthscales = params
    diffs = (x1[:, None] / lengthscales) - (x2[None, :] / lengthscales)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

# Geneate training data corresponding to one input sample
def generate_one_training_data_torch(gp_sample, m=100, P=1, Q=1000):
    # In the source code of Wang et al., only m collocation points of ODE were used,
    # which imposes too few constraints in LSR. 
    # Therefore, the Q parameter is added to increase the number of collocation points.
    
    N = gp_sample.shape[0]
    X = np.linspace(0, 1, N)[:,None]


    # Create a callable interpolation function  
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = np.linspace(0, 1, m)
    # u = vmap(u_fn, in_axes=(None,0))(0.0, x)
    u = np.array([u_fn(0.0, xi) for xi in x])   # vmap → 列表推导

    # Output sensor locations and measurements
    y_train = np.sort(np.random.rand(P))
    s_train = odeint(u_fn, 0.0, np.hstack((0.0, y_train)))[1:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]

    # Tile inputs
    u_train = np.tile(u, (P,1))

    # training data for the residual
    u_r_train = np.tile(u, (Q, 1))
    x_train = np.linspace(0, 1, Q)
    y_r_train = x_train
    s_r_train = np.array([u_fn(0.0, xi) for xi in x_train]).reshape(-1,1)

    return u_train, y_train.reshape(-1,1), s_train, u_r_train, y_r_train.reshape(-1,1), s_r_train

def generate_one_test_data_torch(P):
    
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10

    X = np.linspace(0, 1, N)[:, None]
    K = RBF_np(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = L @ np.random.randn(N)
    
    # Create a callable interpolation function  
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = np.linspace(0, 1, m)
    # u = vmap(u_fn, in_axes=(None,0))(0.0, x)
    u = np.array([u_fn(0.0, xi) for xi in x])   # vmap → 列表推导

    # Output sensor locations and measurements
    y = np.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)

    # Tile inputs
    u = np.tile(u, (P,1))

    return gp_sample, u, y.reshape(-1,1), s 

# ------------------------ Example usage (loading data) ------------------
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torch.load('TrainedModels/torch_model.pth')
    
    lsr = LSR(model, device, folder_name='A01')
    
    # GRF length scale
    length_scale = 0.2
    
    m = 100
    P_train = 1 # number of output sensors, 100 for each side 
    Q_train = 100  # number of collocation points for each input sample

    P_test = m
    
    ## Used to determine the optimal rank
    # rank_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    # loss_LSR_list = []
    # for rank in rank_list:
    #     lsr.compute_error_LSR(30, rank)
    #     loss_LSR_list.append(np.array(lsr.log["res_LSR"]).mean())
    # for rank, loss in zip(rank_list, loss_LSR_list):
    #     print(f'Rank: {rank} Mean loss_LSR: {loss}')
        
    
    ## Compute relative l2 error (LSR)
    lsr.compute_error_LSR(300, rank_LSR=40)
    avg_dict = {key: np.array(value).mean() for key, value in lsr.log.items()}
    print(avg_dict)

    std_dict = {key: np.array(value).std() for key, value in lsr.log.items()}
    print(std_dict)
    
    # torch.save(lsr.log, 'Result/result.pt')

