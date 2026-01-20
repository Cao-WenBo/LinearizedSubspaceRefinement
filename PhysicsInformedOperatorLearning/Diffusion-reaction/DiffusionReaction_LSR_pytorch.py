import numpy as np
import torch
from torch.func import vmap, jacrev, hessian, jvp, vjp, functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
import pickle
import os 
import shutil
import time
import copy
import scipy

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
            DU_partial = DU[:,0:1] # only ddx
            return DU_partial, (U, DU)
        
        def forward_DDU(u, y):
            DDU_partial, (U, DU) = jacrev(forward_DU, argnums=1, has_aux=True)(u, y)
            return U, DU, DDU_partial
        
        U, DU, DDU_partial = vmap(forward_DDU)(self.u_batch, self.y_batch)
        DDU_partial = DDU_partial.squeeze(1)
            
        [U_bc, U_pde] = torch.split(U, self.split_sizes)
        [DU_bc, DU_pde] = torch.split(DU, self.split_sizes)
        [_, DDU_partial_pde] = torch.split(DDU_partial, self.split_sizes)

        [s_ic, s_pde] = torch.split(self.s_batch, self.split_sizes)
        
        r_bc = U_bc - s_ic
        
        s = U_pde
        s_x = DU_pde[:,0:1,0]
        s_t = DU_pde[:,0:1,1]
        s_xx = DDU_partial_pde[:,0:1,0]

        r_pde = s_t - 0.01 * s_xx - 0.01 * s**2 - s_pde
        
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
        chunk_size = 300

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

        end_mem = torch.cuda.memory_allocated()
        end_peak = torch.cuda.max_memory_allocated()

        print(f"Memory: {end_peak / 1024**3:.2f} GB", end=' ')
        
    def compute_error_LSR(self, N_test, rank_LSR):
        
        self.log = {'error':[],'error_LSR':[],'res':[],'res_LSR':[],'time':[]}
        
        for i in range(N_test):

            gp_sample, u_test, y_test, s_test = generate_one_test_data_torch(P)
            
            u_train, y_train, s_train, u_r_train, y_r_train, s_r_train =  \
                generate_one_training_data_torch(gp_sample, P=P_train, Q=Q_train)
            
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

def solve_ADR_np(gp_sample, Nx, Nt, P, length_scale):
    xmin, xmax = 0., 1.
    tmin, tmax = 0., 1.

    k_fn = lambda x: 0.01 * np.ones_like(x)
    v_fn = lambda x: np.zeros_like(x)
    g_fn = lambda u: 0.01 * u ** 2
    dg_fn = lambda u: 0.02 * u
    u0_fn = lambda x: np.zeros_like(x)

    # === GP sample ===
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10

    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF_np(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))


    f_fn = lambda xx: np.interp(xx, X.flatten(), gp_sample)

    # === Grid ===
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h * h

    k = k_fn(x)
    v = v_fn(x)
    f = f_fn(x)

    # Operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)

    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + \
             2 * h * np.diag(v[2:] - v[:Nx - 2])
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0_fn(x)

    for i in range(Nt - 1):
        gi = g_fn(u[1:-1, i])
        dgi = dg_fn(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i]
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)

    xx = np.linspace(xmin, xmax, m)
    u_in = f_fn(xx)

    idx = np.stack([
        np.random.randint(0, Nx, size=P),
        np.random.randint(0, Nt, size=P)
    ], axis=1)

    y = np.hstack([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]])
    s = u[idx[:, 0], idx[:, 1]]

    return (x, t, u), (u_in, y, s)

def generate_one_training_data_torch(gp_sample, P, Q):
    
    (x, t, UU), (u, y, s) = solve_ADR_np(gp_sample, Nx, Nt, P, length_scale)

    # BC/IC points
    x_bc1 = np.zeros((P//3,1))
    x_bc2 = np.ones((P//3,1))
    x_bc3 = np.random.uniform(0,1,(P//3,1))
    x_bcs = np.vstack([x_bc1, x_bc2, x_bc3])

    t_bc1 = np.random.uniform(0,1,(2*P//3,1))
    t_bc2 = np.zeros((P//3,1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    # BC/IC training data
    u_train = np.tile(u, (P,1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P,1))

    # # Collocation points
    # x_r_idx = np.random.choice(np.arange(Nx), size=(Q,1))
    # x_r = x[x_r_idx]
    # t_r = np.random.uniform(0,1,(Q,1))
    # u_r_train = np.tile(u, (Q,1))
    # y_r_train = np.hstack([x_r, t_r])
    # s_r_train = u[x_r_idx]

    # Collocation points
    # === GP sample ===
    N = 512
    X = np.linspace(0, 1, N)[:, None]
    f_fn = lambda xx: np.interp(xx, X.flatten(), gp_sample)
    
    x_r = np.random.uniform(0,1,(Q,1))
    t_r = np.random.uniform(0,1,(Q,1))
    u_r_train = np.tile(u, (Q,1))
    y_r_train = np.hstack([x_r, t_r])
    s_r_train = f_fn(x_r)

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train

def generate_one_test_data_torch(P):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10

    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF_np(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = L @ np.random.randn(N)

    Nx = P
    Nt = P

    ## For verifying grid convergence, k=1 is insufficient to obtain a solution with sufficiently high accuracy.
    # UU_list = []
    # for k in range(1,6):
    #     (x, t, UU), (u, y, s) = solve_ADR_np(gp_sample, k*(Nx-1)+1, k*(Nt-1)+1, k*(m-1)+1, P)
    #     UU_list.append(UU[::k,::k])
    # UU_arr = np.array(UU_list)
    # print(np.abs(UU_arr[1:]-UU_arr[:-1]).mean(1).mean(1))

    k = 4
    (x, t, UU), (u, y, s) = solve_ADR_np(gp_sample, k*(Nx-1)+1, k*(Nt-1)+1, k*(m-1)+1, P)
    x = x[::k]
    t = t[::k]
    UU = UU[::k, ::k]
    u = u


    XX, TT = np.meshgrid(x, t)
    u_test = np.tile(u, (P**2,1))
    y_test = np.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.reshape(-1,1)

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
    
    N = 5000 # number of input samples
    m = Nx   # number of input sensors
    P_train = 3000 # number of output sensors, 100 for each side 
    Q_train = 20000  # number of collocation points for each input sample

    P = m

    ## Search for the rank with the smallest loss.
    # rank_list = [50, 80, 100, 150, 200, 250, 300]
    # loss_LSR_list = []
    # error_LSR_list = []
    # for rank in rank_list:
    #     lsr.compute_error_LSR(30, rank)
    #     loss_LSR_list.append(np.array(lsr.log["res_LSR"]).mean())
    #     error_LSR_list.append(np.array(lsr.log["error_LSR"]).mean())
    # for rank, loss, error in zip(rank_list, loss_LSR_list, error_LSR_list):
    #     print(f'Rank: {rank} Mean loss_LSR: {loss} Mean error_LSR: {error}')
        
    
    ## Compute relative l2 error (LSR)
    lsr.compute_error_LSR(300, rank_LSR=200)
    avg_dict = {key: np.array(value).mean() for key, value in lsr.log.items()}
    print(avg_dict)

    std_dict = {key: np.array(value).std() for key, value in lsr.log.items()}
    print(std_dict)
    
    torch.save(lsr.log, 'Result/result.pt')
