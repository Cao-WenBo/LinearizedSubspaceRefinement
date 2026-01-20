import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.func import vmap, jacrev, hessian, jvp, vjp, functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
import pickle
import os 
import shutil
import copy
from itertools import islice
from CLS_MNIST import ConvClassifier
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def softmax_torch(x):
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

def vector_to_param_dict(model, theta_vec):
    param_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param_dict[name] = theta_vec[pointer:pointer + numel].view_as(param)
        pointer += numel
    return param_dict

class LSR():

    def __init__(self, trainloader, testloader, model, device):
        self.device = device
        self.model = model
        
        self.trainloader, self.testloader = trainloader, testloader

    def forward(self, params_vector, X):
        params_dict = vector_to_param_dict(self.model, params_vector)
        f = functional_call(self.model, params_dict, X)
        f = softmax_torch(f)
        return f

    def System(self, params_vector):
        X = self.X_batch

        U = self.forward(params_vector, X)
        
        R = (U - self.Y_batch).reshape(-1,1)
        return R
    
    def LSR(self, rank=1000):
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()
        
        oversample = 10
        k = rank + oversample
        chunk_size = 32  # This parameter depends on the size of the GPU memory.

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
            for self.X_batch, self.Y_batch in islice(self.trainloader, 10):
                self.X_batch, self.Y_batch = self.X_batch.to(self.device), one_hot(self.Y_batch.to(self.device), num_classes=10)
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
                self.X_batch, self.Y_batch = self.X_batch.to(self.device), one_hot(self.Y_batch.to(self.device), num_classes=10)

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

    def train(self, rank_LSR):

        ## LSR
        self.params_vector_base = parameters_to_vector(self.model.parameters())
        t1 = time.time()
        self.LSR(rank=rank_LSR)
        t2 = time.time()
        print('time:', t2-t1, end=' ')

        correct = 0
        correct_LSR = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                f0, f_linear = jvp(lambda params: self.forward(params, imgs), (self.params_vector_base,), (self.params_vector_delta,))
                pred = f0.argmax(dim=1)
                pred_LSR = (f0 + f_linear).argmax(dim=1)
                correct += (pred == labels).sum().item()
                correct_LSR += (pred_LSR == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        acc_LSR = correct_LSR / total
        print(f"Train Accuracy: {acc:.4f} Train Accuracy (LSR): {acc_LSR:.4f}", end=' ')

        correct = 0
        correct_LSR = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.testloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                f0, f_linear = jvp(lambda params: self.forward(params, imgs), (self.params_vector_base,), (self.params_vector_delta,))
                pred = f0.argmax(dim=1)
                pred_LSR = (f0 + f_linear).argmax(dim=1)
                correct += (pred == labels).sum().item()
                correct_LSR += (pred_LSR == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        acc_LSR = correct_LSR / total
        print(f"Test Accuracy: {acc:.4f} Test Accuracy (LSR): {acc_LSR:.4f}")
        
    def Pred(self, X):
        f0, f_linear = jvp(lambda params: self.forward(params, X), (self.params_vector_base,), (self.params_vector_delta,))
        return f0 + f_linear



if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if 1 else "cpu")
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    class Config:
        batch_size = 128
        num_epochs = 50
        lr = 1e-3
        weight_decay = 1e-5
        latent_dim = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        activation = "gelu"
        save_dir = "./results_classifier"

    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    model = ConvClassifier(cfg.latent_dim, cfg.activation).to(cfg.device)
    
    batch_size = 2048 # batch_size * n_class must > rank
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    lsr = LSR(trainloader, testloader, model, device)
    
    for rank_LSR in [2] + list(range(1000, 21000, 1000)):
        lsr.train(rank_LSR)
        
        

    


