import numpy as np
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
import scipy

from JaxModel2TorchModel import TorchModifiedMLP, TorchPI_DeepONet

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
        
        s = U_pde
        s_x = DU_pde[:,0:1,0]
        s_t = DU_pde[:,0:1,1]

        r_pde = s_t + self.ux_batch*s_x
        
        N_bc = r_bc.shape[0]
        N_pde = r_pde.shape[0]
        N_tot = N_bc + N_pde
    
        # exact weights
        w_bc = (100 * N_tot / N_bc)**0.5
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
        chunk_size = 64

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
            self.u_batch0, self.y_batch0 = self.u_batch * 1, self.y_batch * 1
            rand_ind = torch.randperm(self.u_batch.shape[0])[:(rank+1000)]
            self.u_batch, self.y_batch = self.u_batch[rand_ind], self.y_batch[rand_ind] #Use a small dataset to determine V.

            JO = JV_fn(Omega)
            Q, R = torch.linalg.qr(JO, mode='reduced')
            JTQ = JTV_fn(Q)

            U, S, Vh = torch.linalg.svd(JTQ.T, full_matrices=False)
            V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

            self.u_batch, self.y_batch = self.u_batch0, self.y_batch0
            AJV = AJV_fn(V)
            # AJV_inv = torch.pinverse(AJV)
            # AJV_inv = torch.linalg.pinv(AJV, rtol=1e-15)
            # self.params_vector_delta = (V@(AJV_inv@(-f0))).flatten()
            self.params_vector_delta = (V@torch.linalg.lstsq(AJV, -f0, rcond=1e-15).solution).flatten()
            # print(f"Cond: {torch.linalg.cond(AJV).item():.0f}", end=' ')  

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

        end_mem = torch.cuda.memory_allocated()
        end_peak = torch.cuda.max_memory_allocated()

        print(f"Memory: {end_peak / 1024**3:.2f} GB", end=' ')
        
    def compute_error_LSR(self, N_test, rank_LSR):
        
        self.log = {'error':[],'error_LSR':[],'res':[],'res_LSR':[],'time':[]}
        
        for i in range(N_test):

            gp_sample, u_test, y_test, s_test = generate_one_test_data_torch(Nx=100, Nt=100, P=100)
            
            u_train, y_train, s_train, u_r_train, y_r_train, s_r_train =  \
                generate_one_training_data_torch(gp_sample, P=P_train, Q=Q_train)
            
            u_train, y_train, s_train, u_r_train, y_r_train, s_r_train =  \
                torch.tensor(u_train, device=self.device).float(), torch.tensor(y_train, device=self.device).float(), \
                torch.tensor(s_train, device=self.device).float(), torch.tensor(u_r_train, device=self.device).float(), \
                torch.tensor(y_r_train, device=self.device).float(), torch.tensor(s_r_train, device=self.device).float()
                
            u_test, y_test, s_test = torch.tensor(u_test, device=self.device).float(), \
                torch.tensor(y_test, device=self.device).float(), torch.tensor(s_test, device=self.device).float()
            
            self.u_batch = torch.cat([u_train, u_r_train])
            self.y_batch = torch.cat([y_train, y_r_train[:,:2]])
            self.s_batch = torch.cat([s_train, s_r_train])
            
            self.ux_batch = y_r_train[:,2:3]
            
            self.split_sizes = [u_train.shape[0], u_r_train.shape[0]]
            
            
            self.params_vector_base = parameters_to_vector(self.model.parameters()).detach().clone()
            
            t1 = time.perf_counter()
            self.LSR(rank=rank_LSR)
            t2 = time.perf_counter()
            
    
            f0, f_linear = jvp(lambda params: self.forward(params, u_test, y_test), (self.params_vector_base,), (self.params_vector_delta,))
            pred = f0
            pred_LSR = f0 + f_linear
            error = torch.linalg.norm(pred - s_test) / torch.linalg.norm(s_test)
            error_LSR = torch.linalg.norm(pred_LSR - s_test) / torch.linalg.norm(s_test)
    

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

f_np = lambda x: np.sin(np.pi * x)
g_np = lambda t: np.sin(np.pi * t / 2)


def solve_CVC_np(gp_sample, Nx, Nt, m, P):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1

    N = gp_sample.shape[0]
    X = np.linspace(xmin, xmax, N)
    V_fn = lambda x: np.interp(x, X, gp_sample)

    # Grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # Velocity field
    v_fn = lambda x: V_fn(x) - np.min(gp_sample) + 1.0
    v = v_fn(x)

    # Initialize solution
    u = np.zeros((Nx, Nt))

    # Boundary conditions
    u[0, :] = g_np(t)
    u[:, 0] = f_np(x)

    # Finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)

    # Build matrices K and D
    K = np.eye(Nx - 1)
    K_temp = np.eye(Nx - 1)
    Trans = np.eye(Nx - 1, k=-1)

    for _ in range(Nx - 2):
        K_temp = -k[:, None] * (Trans @ K_temp)
        K += K_temp

    D = np.diag(k) + np.eye(Nx - 1, k=-1)

    # Time stepping
    for i in range(Nt - 1):
        b = np.zeros(Nx - 1)
        b[0] = g_np(t[i]) - k[0] * g_np(t[i + 1])
        u[1:, i + 1] = K @ (D @ u[1:, i] + b)

    UU = u

    # Sensor input locations
    xx = np.linspace(xmin, xmax, m)
    u_sensor = v_fn(xx)

    # Random output sensors
    idx_x = np.random.randint(0, Nx, size=P)
    idx_t = np.random.randint(0, Nt, size=P)

    y = np.stack([x[idx_x], t[idx_t]], axis=1)
    s = UU[idx_x, idx_t]

    return (x, t, UU), (u_sensor, y, s)

def generate_one_training_data_torch(gp_sample, P, Q):

    N = 512

    X = np.linspace(xmin, xmax, N)[:, None]

    # use input gp_sample
    # velocity
    v_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    u_fn = lambda x: v_fn(x) - v_fn(x).min() + 1.0

    (x, t, UU), (u, y, s) = solve_CVC_np(gp_sample, Nx, Nt, m, P)

    x_bc1 = np.zeros((P // 2, 1))
    x_bc2 = np.random.rand(P // 2, 1)
    x_bcs = np.vstack((x_bc1, x_bc2))

    t_bc1 = np.random.rand(P // 2, 1)
    t_bc2 = np.zeros((P // 2, 1))
    t_bcs = np.vstack((t_bc1, t_bc2))

    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])

    s_bc1 = np.sin(np.pi * t_bc1 / 2)      # g(t)
    s_bc2 = np.sin(np.pi * x_bc2)          # f(x)
    s_train = np.vstack((s_bc1, s_bc2))


    x_r = np.random.uniform(xmin, xmax, size=(Q, 1))
    t_r = np.random.uniform(tmin, tmax, size=(Q, 1))
    ux_r = u_fn(x_r)

    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r, ux_r])
    s_r_train = np.zeros((Q, 1))

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train

def generate_one_test_data_torch(Nx, Nt, P):

    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10

    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF_np(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = L @ np.random.randn(N)


    ## For verifying grid convergence, k=1 is insufficient to obtain a solution with sufficiently high accuracy. 
    # UU_list = []
    # for k in range(1,6):
    #     (x, t, UU), (u, y, s) = solve_CVC_np(gp_sample, k*(Nx-1)+1, k*(Nt-1)+1, k*(m-1)+1, P)
    #     UU_list.append(UU[::k,::k])
    # UU_arr = np.array(UU_list)
    # print(np.abs(UU_arr[1:]-UU_arr[:-1]).mean(1).mean(1))

    k = 5
    (x, t, UU), (u, y, s) = solve_CVC_np(gp_sample, k*(Nx-1)+1, k*(Nt-1)+1, k*(m-1)+1, P)
    x = x[::k]
    t = t[::k]
    UU = UU[::k, ::k]
    u = u[::k]

    XX, TT = np.meshgrid(x, t, indexing='ij')

    u_test = np.tile(u, (Nx * Nt, 1))
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = UU.reshape(-1, 1)

    return gp_sample, u_test, y_test, s_test

# ------------------------ Example usage (loading data) ------------------
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torch.load('TrainedModels/torch_model.pth')
    
    lsr = LSR(model, device, folder_name='A01')
    
    # GRF length scale
    length_scale = 0.2
    
    # Resolution of the solution
    Nx = 100
    Nt = 100
    
    # Computational domain
    xmin = 0.0
    xmax = 1.0
    
    tmin = 0.0
    tmax = 1.0

    P_train = 1000   # number of output sensors, 100 for each side 
    Q_train = 20000  # number of collocation points for each input sample
    rank_LSR = 1000
    
    N = 1000 # number of input samples
    m = Nx   # number of input sensors

    ## Search for the rank with the smallest loss.
    # rank_list = [500, 800, 1000, 1200, 1500, 1800, 2000]
    # loss_LSR_list = []
    # error_LSR_list = []
    # for rank in rank_list:
    #     lsr.compute_error_LSR(30, rank)
    #     loss_LSR_list.append(np.array(lsr.log["res_LSR"]).mean())
    #     error_LSR_list.append(np.array(lsr.log["error_LSR"]).mean())
    # for rank, loss, error in zip(rank_list, loss_LSR_list, error_LSR_list):
    #     print(f'Rank: {rank} Mean loss_LSR: {loss} Mean error_LSR: {error}')
        
    
    ## Compute relative l2 error (LSR)
    lsr.compute_error_LSR(300, rank_LSR=1000)
    avg_dict = {key: np.array(value).mean() for key, value in lsr.log.items()}
    print(avg_dict)

    std_dict = {key: np.array(value).std() for key, value in lsr.log.items()}
    print(std_dict)
    
    torch.save(lsr.log, 'Result/result.pt')
