import sys 
sys.path.append("../..") 
import h5py

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import vmap, jacrev, hessian, jvp, vjp, functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
import pickle
import os 
import shutil
import time
import copy
from torch.utils.data import TensorDataset, DataLoader

def parameters_to_vector(model, p=1.0):
    # Sampling proportion p: 0<p<=1

    vec_list = []
    meta = []      # [(name, para0, mask), ...]

    total_numel = 0
    for name, para in model.named_parameters():
        total_numel += para.numel()

    k = int(total_numel * p)

    global_mask = torch.zeros(total_numel, dtype=torch.bool, device=next(model.parameters()).device)
    perm = torch.randperm(total_numel, device=global_mask.device)[:k]
    global_mask[perm] = True

    pointer = 0
    for name, para in model.named_parameters():
        numel = para.numel()

        local_mask = global_mask[pointer:pointer+numel].reshape(para.shape)
        pointer += numel

        real = para.detach()[local_mask]
        vec_list.append(real.reshape(-1))

        meta.append((name, para, local_mask))

    # 拼接成向量
    theta_vec_sampled = torch.cat(vec_list)

    return theta_vec_sampled, meta

def vector_to_param_dict(vec, meta):
    """
    Args:
        vec: [total_vec_length] 或 [batch, total_vec_length]
        meta: [(name, para0, mask), ...]
    Returns:
        param_dict: {name: tensor}
    """
    out = {}
    ptr = 0

    batch_mode = (vec.dim() == 2)
    batch_size = vec.shape[0] if batch_mode else 1

    for (name, para0, mask) in meta:
        shape = para0.shape
        mask_flat = mask.flatten()
        K = mask_flat.sum()


        real_flat = vec[:, ptr:ptr + K] if batch_mode else vec[ptr:ptr + K]
        ptr += K
        if batch_mode:
            para = para0.unsqueeze(0).expand(batch_size, *para0.shape).clone().to(real_flat.dtype)
            para_flat = para.flatten(1)
            idxs = mask_flat.nonzero(as_tuple=False).squeeze(-1)
            para_flat.scatter_(1, idxs.unsqueeze(0).expand(batch_size, -1), real_flat)
            para = para_flat.view(batch_size, *para0.shape)
        else:
            para_flat = para0.flatten().clone().to(real_flat.dtype)
            idxs = mask_flat.nonzero(as_tuple=False).squeeze(-1)
            para_flat.scatter_(0, idxs, real_flat)
            para = para_flat.view_as(para0)

        out[name] = para

    return out

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class LSR():

    def __init__(self, model, ax_train, u_train, ax_test, u_test, gridx_train):
        self.device = ax_train.device
        self.model = model

        batch_size = 160 # The product of batch_size and out_dim must be larger than rank_LSR.
        self.trainloader = DataLoader(TensorDataset(ax_train, u_train), batch_size=batch_size, shuffle=True)
        self.X_test, self.Y_test = ax_test, u_test
        self.gridx_train = gridx_train


    def forward(self, params_vector, X):
        params_dict = vector_to_param_dict(params_vector, self.meta)
        f = functional_call(self.model, params_dict, (self.gridx_train, X))
        return f

    def System(self, params_vector):
        X = self.X_batch
        def forward_U(X):
            U = self.forward(params_vector, X)
            return U
        
        U = forward_U(X)
        U = U.squeeze(1)
        
        R = (U - self.Y_batch).reshape(-1,1)
        return R
    
    def LSR(self, rank=1000):
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()
        
        oversample = 10
        k = rank + oversample
        chunk_size = 256  # This parameter depends on the size of the GPU memory.

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
            
            Omega = torch.randn(self.params_vector_base.shape[0], k, device=self.device)

            JTQ = torch.zeros(self.params_vector_base.shape[0], k, device=self.device)
            for self.X_batch, self.Y_batch in self.trainloader:

                JO = JV_fn(Omega)
                Q, R = torch.linalg.qr(JO, mode='reduced')
                if Q.shape[1] != JO.shape[1]: continue # Discard the last batch if its size is insufficient for QR decomposition.

                JTQ += JTV_fn(Q)
                del JO, Q, R

            U, S, Vh = torch.linalg.svd(JTQ.T, full_matrices=False)
            V = (Vh.T@torch.diag(1 / S))[:,:rank]
            # V = (Vh.T)[:,:rank]

            del Omega,  JTQ, U, S, Vh
            torch.cuda.empty_cache()

            ## batch QR
            R = torch.zeros(V.shape[1], V.shape[1], device=device)
            z = torch.zeros(V.shape[1], 1, device=device)

            for self.X_batch, self.Y_batch in self.trainloader:

                A_batch = JV_fn(V)         
                b_batch = -self.System(self.params_vector_base)

                M = torch.cat([R, A_batch], dim=0)

                Qm, R_new = torch.linalg.qr(M, mode='reduced')
                R = R_new

                zb = torch.cat([z, b_batch], dim=0) 
                z = Qm.T @ zb     
                del A_batch, b_batch, M, Qm

            self.params_vector_delta = (V@torch.linalg.solve(R, z)).flatten()
            torch.cuda.empty_cache()

            end_mem = torch.cuda.memory_allocated()
            end_peak = torch.cuda.max_memory_allocated()

            print(f"Memory: {end_peak / 1024**3:.2f} GB", end=' ')
            # print(f"Cond: {torch.linalg.cond(JV).item():.0f}", end=' ')  

        
    def train(self):

        ## LSR
        self.params_vector_base, self.meta = parameters_to_vector(self.model, p=0.5)

        t1 = time.time()
        self.LSR(rank=rank_LSR)
        t2 = time.time()
        print(f'rank:{rank_LSR} time:{t2-t1:4g}', end=' ')


        f0 = []; f_linear = []
        for self.X_batch, self.Y_batch in self.trainloader:
            f0_batch, f_linear_batch = jvp(lambda params: self.System(params), (self.params_vector_base,), (self.params_vector_delta,))
            f0.append(f0_batch)
            f_linear.append(f_linear_batch)
        f0 = torch.cat(f0)
        f_linear = torch.cat(f_linear)
        self.loss = (f0**2).mean().item()
        self.loss_LSR = ((f0 + f_linear)**2).mean().item()
        print(f'Loss: {self.loss:6g} Loss (LSR) {self.loss_LSR:6g}', end=' ')

        f0, f_linear = jvp(lambda params: self.forward(params, self.X_test), (self.params_vector_base,), (self.params_vector_delta,))
        pred = f0
        
        diff_norms = torch.norm(pred.reshape(pred.shape[0],-1) - self.Y_test.reshape(pred.shape[0],-1), 2, 1)
        y_norms = torch.norm(self.Y_test.reshape(pred.shape[0],-1), 2, 1) 
        #
        self.error_mean = torch.mean(diff_norms/y_norms).item()
        self.error_std = torch.std(diff_norms/y_norms).item()

        
        pred = f0 + f_linear
        diff_norms = torch.norm(pred.reshape(pred.shape[0],-1) - self.Y_test.reshape(pred.shape[0],-1), 2, 1)
        y_norms = torch.norm(self.Y_test.reshape(pred.shape[0],-1), 2, 1) 
        #
        self.error_mean_LSR = torch.mean(diff_norms/y_norms).item()
        self.error_std_LSR = torch.std(diff_norms/y_norms).item()

        print(f'Error mean: {self.error_mean:6g} Error std: {self.error_std:6g} Error mean (LSR): {self.error_mean_LSR:6g} Error std (LSR): {self.error_std_LSR:6g}')
        
    def Pred(self, X):
        f0, f_linear = jvp(lambda params: self.forward(params, X), (self.params_vector_base,), (self.params_vector_delta,))
        return f0 + f_linear

        
# %%
if __name__ == '__main__':
    device = 'cuda:0'
    dtype = torch.float32


    data_train = h5py.File('../Data/Burgers_1d/viscid_train.mat', 'r')
    data_test = h5py.File('../Data/Burgers_1d/viscid_test_in.mat', 'r')


    from Utils.utils import *
    n_train, n_test = 1000, 50
    def get_data(data, ndata, dtype, n0=0):
        # Data is of the shape (number of samples = 1000, grid size = 29*29)
        a = np2tensor(np.array(data["u0"][...,n0:n0+ndata]).T, dtype)
        u = np2tensor(np.array(data["u_sol"][...,n0:n0+ndata]).T, dtype)
        uT = u[:,-1,:]
        x_mesh = np2tensor(np.array(data['x_mesh']))
        #
        a = a.reshape(ndata, -1)
        uT = uT.reshape(ndata, -1, 1)
    
        return a, uT, x_mesh
    
    a_train, uT_train, gridx_train = get_data(data_train, n_train, dtype)
    a_test, uT_test, gridx_test = get_data(data_test, n_test, dtype)
    
    # gridx_train == gridx_test
    a_train, uT_train, a_test, uT_test, gridx_train = a_train.to(device), \
        uT_train.to(device), a_test.to(device), uT_test.to(device), gridx_train.to(device)
    
    # %%
    from Solvers.MultiONet import MultiONet
    solver = MultiONet.Solver(device, dtype)
    netType = 'MultiONetCartesian_Tanh'
    model_trained = solver.loadModel(path=f'saved_models/{netType}/', name='model_multionet_final')
    
    model = model_trained['u']
    
    lsr = LSR(model, a_train, uT_train, a_test, uT_test, gridx_train)
    result = {'loss':[], 'loss_LSR':[], 'error_mean':[], 'error_std':[], 'error_mean_LSR':[], 'error_std_LSR':[], 'params_vector_delta':[], 'meta':[]}

    for rank_LSR in range(1000, 21000, 1000):
        try: ##Cuda out of memory
            lsr.train()
        except:
            break
        result['loss'].append(lsr.loss)
        result['loss_LSR'].append(lsr.loss_LSR)
        result['error_mean'].append(lsr.error_mean)
        result['error_std'].append(lsr.error_std)
        result['error_mean_LSR'].append(lsr.error_mean_LSR)
        result['error_std_LSR'].append(lsr.error_std_LSR)
        result['params_vector_delta'].append(lsr.params_vector_delta)
        result['meta'].append(lsr.meta)

    argmin = np.argmin(np.array(result['loss_LSR']))
    print(f'min loss_LSR: {result["loss_LSR"][argmin]} error: {result["error_mean"][argmin]} +- {result["error_std"][argmin]}', end=' ')
    print(f'error_LSR: {result["error_mean_LSR"][argmin]} +- {result["error_std_LSR"][argmin]}')

    argmin = np.argmin(np.array(result['loss_LSR']))
    print(f'min loss_LSR: {result["loss_LSR"][argmin]} error: {result["error_mean"][argmin]} +- {result["error_std"][argmin]}', end=' ')
    print(f'error_LSR: {result["error_mean_LSR"][argmin]} +- {result["error_std_LSR"][argmin]}')

    rank_LSR = list(range(1000, 21000, 1000))[argmin]
    lsr.train()
    f0, f_linear = jvp(lambda params: lsr.forward(params, lsr.X_test), (lsr.params_vector_base,), (lsr.params_vector_delta,))
    pred = f0
    pred_lsr = f0 + f_linear
    Result_test_pred = {'X_test':lsr.X_test.cpu().detach().numpy(), 'Y_pred':pred.cpu().detach().numpy(), 'Y_pred_lsr':pred_lsr.cpu().detach().numpy(), 'Y_test':lsr.Y_test.cpu().detach().numpy()}
    torch.save(Result_test_pred, os.path.join(f'saved_models/{netType}/', 'result_testdata_LSR_p0.5.pt'))

    import os
    torch.save(result, os.path.join(f'saved_models/{netType}/', 'result_LSR_p0.5.pt'))

    
    #########################################
    with torch.no_grad():
        uT_pred = model_trained['u'](gridx_test.to(device), a_test)
        uT_pred = uT_pred.detach()
        
        uT_pred_LSR = lsr.Pred(a_test).detach()
    #
    print('\n')
    print('The test loss', solver.getLoss(uT_pred, uT_test))
    print('The test loss (LSR)', solver.getLoss(uT_pred_LSR, uT_test))
    print('The test l2 error:', solver.getError(uT_pred, uT_test))
    print('The test l2 error (LSR):', solver.getError(uT_pred_LSR, uT_test))
    print('*************************************')
    for i in range(0, 5):
        print(f'The test l2 error for {i}:', solver.getError(uT_pred[i:i+1], uT_test[i:i+1]))
        print(f'The test l2 error for {i} (LSR):', solver.getError(uT_pred_LSR[i:i+1], uT_test[i:i+1]))
        
    inx = 0
    uT_pred_LSR = lsr.Pred(a_test[inx:inx+1])[0]
    plt.plot(gridx_test.cpu(), uT_test[inx].cpu())
    plt.plot(gridx_test.cpu(), uT_pred[inx].cpu())
    plt.plot(gridx_test.cpu(), uT_pred_LSR.cpu().detach())
    # inx = 0
    # # ########################################
    # from Utils.PlotFigure import Plot
    # # show prediction
    # Plot.show_1d_list(gridx_test, [uT_test[inx], uT_pred[inx]], 
    #                   label_list=['uT0_True', 'uT0_Pred'], title='prediction of uT')
    # # show loss
    # loss_saved = solver.loadLoss(path=f'saved_models/{netType}/', name='loss_deeponet')
    # Plot.show_loss([loss_saved['loss_train'], loss_saved['loss_test']], ['loss_train', 'loss_test'])
    # # show error
    # Plot.show_error([loss_saved['time']], [loss_saved['l2_test']], ['l2_test'])
        
    
