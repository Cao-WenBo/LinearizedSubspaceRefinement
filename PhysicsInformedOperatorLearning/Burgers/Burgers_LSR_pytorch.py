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

    def System_nonlinear(self, params_vector_delta):

        def forward_U(u, y):
            f0, f_linear = jvp(lambda params: self.forward(params, u, y), (self.params_vector_base,), (params_vector_delta,))
            U = f0 + f_linear
            return U, U
        
        def forward_DU(u, y):
            DU, U = jacrev(forward_U, argnums=1, has_aux=True)(u, y)
            DU_partial = DU[:,1:2] # only ddx
            return DU_partial, (U, DU)
        
        def forward_DDU(u, y):
            DDU_partial, (U, DU) = jacrev(forward_DU, argnums=1, has_aux=True)(u, y)
            return U, DU, DDU_partial
        
        U, DU, DDU_partial = vmap(forward_DDU)(self.u_batch, self.y_batch)
        DDU_partial = DDU_partial.squeeze(1)
            
        [U_ic, U_lbc, U_ubc, U_pde] = torch.split(U, self.split_sizes)
        [DU_ic, DU_lbc, DU_ubc, DU_pde] = torch.split(DU, self.split_sizes)
        [_, _, _, DDU_partial_pde] = torch.split(DDU_partial, self.split_sizes)
        
        [s_ic, s_lbc, s_ubc, s_pde] = torch.split(self.s_batch, self.split_sizes)
        
        r_ic = U_ic - s_ic
        
        r_bc1 = U_ubc - U_lbc
        r_bc2 = DU_ubc[:,0:1,1] - DU_lbc[:,0:1,1]
        
        u = U_pde
        u_x = DU_pde[:,0:1,1]
        u_t = DU_pde[:,0:1,0]
        u_xx = DDU_partial_pde[:,0:1,1]

        r_pde = u_t + u*u_x - 0.01*u_xx
        

        N_ic = r_ic.shape[0]
        N_bc = r_bc1.shape[0]
        N_pde = r_pde.shape[0]
        N_tot = N_ic + N_bc + N_bc + N_pde
    
        # exact weights
        w_ic = (20 * N_tot / N_ic)**0.5
        w_bc = (1 * N_tot / N_bc)**0.5
        w_pde = (1 * N_tot / N_pde)**0.5
    
        RR = torch.cat([w_ic*r_ic, w_bc*r_bc1, w_bc*r_bc2, w_pde*r_pde])
        return RR

    def System(self, params_vector):

        def forward_U(u, y):
            U = self.forward(params_vector, u, y)
            return U, U
        
        def forward_DU(u, y):
            DU, U = jacrev(forward_U, argnums=1, has_aux=True)(u, y)
            DU_partial = DU[:,1:2] # only ddx
            return DU_partial, (U, DU)
        
        def forward_DDU(u, y):
            DDU_partial, (U, DU) = jacrev(forward_DU, argnums=1, has_aux=True)(u, y)
            return U, DU, DDU_partial
        
        U, DU, DDU_partial = vmap(forward_DDU)(self.u_batch, self.y_batch)
        DDU_partial = DDU_partial.squeeze(1)
            
        [U_ic, U_lbc, U_ubc, U_pde] = torch.split(U, self.split_sizes)
        [DU_ic, DU_lbc, DU_ubc, DU_pde] = torch.split(DU, self.split_sizes)
        [_, _, _, DDU_partial_pde] = torch.split(DDU_partial, self.split_sizes)
        
        [s_ic, s_lbc, s_ubc, s_pde] = torch.split(self.s_batch, self.split_sizes)
        
        r_ic = U_ic - s_ic
        
        r_bc1 = U_ubc - U_lbc
        r_bc2 = DU_ubc[:,0:1,1] - DU_lbc[:,0:1,1]
        
        u = U_pde
        u_x = DU_pde[:,0:1,1]
        u_t = DU_pde[:,0:1,0]
        u_xx = DDU_partial_pde[:,0:1,1]

        r_pde = u_t + u*u_x - 0.01*u_xx
        

        N_ic = r_ic.shape[0]
        N_bc = r_bc1.shape[0]
        N_pde = r_pde.shape[0]
        N_tot = N_ic + N_bc + N_bc + N_pde
    
        # exact weights
        w_ic = (20 * N_tot / N_ic)**0.5
        w_bc = (1 * N_tot / N_bc)**0.5
        w_pde = (1 * N_tot / N_pde)**0.5
    
        RR = torch.cat([w_ic*r_ic, w_bc*r_bc1, w_bc*r_bc2, w_pde*r_pde])
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

            del Omega, JO, Q, R, JTQ, U, S, Vh

            self.u_batch, self.y_batch = self.u_batch0, self.y_batch0

            AJV = AJV_fn(V)
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
        
    def compute_error_LSR(self, usol, rank_LSR):
        
        self.log = {'error':[],'error_LSR':[],'res':[],'res_LSR':[],'res_nonlinear_LSR':[],'time':[]}
        
        for i in range(usol.shape[0]):
            u0 = usol[i,0:1]
            
            u_ics_train, y_ics_train, s_ics_train = generate_one_ics_training_data_torch(u0, 101, 101)
            u_bcs_train, y_bcs_train, s_bcs_train = generate_one_bcs_training_data_torch(u0, 101, 1000)
            u_res_train, y_res_train, s_res_train = generate_one_res_training_data_torch(u0, 101, 10000)
            
            u_test, y_test, s_test = generate_one_test_data(usol[i])
            
            self.u_batch = torch.cat([u_ics_train, u_bcs_train, u_bcs_train, u_res_train])
            self.y_batch = torch.cat([y_ics_train, y_bcs_train[:,0:2], y_bcs_train[:,2:4], y_res_train]) # left and right
            self.s_batch = torch.cat([s_ics_train, s_bcs_train, s_bcs_train, s_res_train])
            
            self.split_sizes = [u_ics_train.shape[0], u_bcs_train.shape[0], u_bcs_train.shape[0], u_res_train.shape[0]]
            
            
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

            f = self.System_nonlinear(self.params_vector_delta)
            res_nonlinear_LSR = ((f)**2).mean()

    
            self.log['error'].append(error.item())
            self.log['error_LSR'].append(error_LSR.item())
            self.log['res'].append(res.item())
            self.log['res_LSR'].append(res_LSR.item())
            self.log['res_nonlinear_LSR'].append(res_nonlinear_LSR.item())
            self.log['time'].append(t2-t1)
            
            print(f'{i}|{usol.shape[0]} error={error.item()} error_LSR={error_LSR.item()} res={res.item()} res_LSR={res_LSR.item()} time={t2-t1}')

def generate_one_ics_training_data_torch(u0, m=101, P=101):
    t_0 = torch.zeros((P, 1), device=u0.device)
    x_0 = torch.linspace(0, 1, P, device=u0.device).unsqueeze(1)

    y = torch.cat([t_0, x_0], dim=1)            # (P, 2)
    u = u0.repeat(P, 1)            # (P, m)
    s = u0.T                              # (m,1)

    return u, y, s

def generate_one_bcs_training_data_torch(u0, m=101, P=100):
    t_bc = torch.rand((P, 1), device=u0.device)
    x_bc1 = torch.zeros((P, 1), device=u0.device)
    x_bc2 = torch.ones((P, 1), device=u0.device)

    y1 = torch.cat([t_bc, x_bc1], dim=1)        # (P, 2)
    y2 = torch.cat([t_bc, x_bc2], dim=1)        # (P, 2)

    y = torch.cat([y1, y2], dim=1)              # (P, 4)
    u = u0.repeat(P, 1)            # (P, m)
    s = torch.zeros((P, 1), device=u0.device)

    return u, y, s

def generate_one_res_training_data_torch(u0, m=101, P=1000):
    t_res = torch.rand((P, 1), device=u0.device)
    x_res = torch.rand((P, 1), device=u0.device)

    y = torch.cat([t_res, x_res], dim=1)        # (P, 2)
    u = u0.repeat(P, 1)            # (P, m)
    s = torch.zeros((P, 1), device=u0.device)

    return u, y, s

def generate_one_test_data(u, m=101, P=101):
    u0 = u[0:1]
    t = torch.linspace(0.0, 1.0, P, device=u0.device)
    x = torch.linspace(0.0, 1.0, P, device=u0.device)
    T, X = torch.meshgrid(t,x)
    
    s = u.reshape(-1,1)
    u_tile = u0.repeat(P**2, 1)
    y = torch.cat([T.reshape(-1,1), X.reshape(-1,1)], dim=1)
    return u_tile, y, s

# ------------------------ Example usage (loading data) ------------------
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = scipy.io.loadmat('Data/Burger.mat')
    usol = torch.tensor(data['output'], device=device).float()
    
    model = torch.load('TrainedModels/torch_model.pth')
    
    lsr = LSR(model, device, folder_name='A01')

    ## Search for the rank with the smallest loss.
    # rank_list = [500, 800, 1000, 1200, 1500, 1800, 2000]
    # loss_LSR_list = []
    # error_LSR_list = []
    # for rank in rank_list:
    #     lsr.compute_error_LSR(usol[-30:], rank)
    #     loss_LSR_list.append(np.array(lsr.log["res_LSR"]).mean())
    #     error_LSR_list.append(np.array(lsr.log["error_LSR"]).mean())
    # for rank, loss, error in zip(rank_list, loss_LSR_list, error_LSR_list):
    #     print(f'Rank: {rank} Mean loss_LSR: {loss} Mean error_LSR: {error}')
        
    
    ## Compute relative l2 error (LSR)
    lsr.compute_error_LSR(usol[-300:], rank_LSR=800)
    avg_dict = {key: np.array(value).mean() for key, value in lsr.log.items()}
    print(avg_dict)

    std_dict = {key: np.array(value).std() for key, value in lsr.log.items()}
    print(std_dict)
    
    torch.save(lsr.log, 'Result/result.pt')

    # with open("A02-rank1000-mode2-10000-1000.pkl", "wb") as f:
    #     pickle.dump(nn.log, f)
        
    # with open("A01-rank2.pkl", "rb") as f:
    #     my_dict = pickle.load(f)
