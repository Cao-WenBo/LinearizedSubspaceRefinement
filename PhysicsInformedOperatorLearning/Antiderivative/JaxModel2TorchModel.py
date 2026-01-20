import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
from jax import random, vmap
from JaxModel import PI_DeepONet

class TorchMLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super(TorchMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation()

        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            self.layers.append(linear)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# Modified MLP
class TorchModifiedMLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        self.activation = activation()
        self.n_layers = len(layers) - 1
        
        # 普通层
        self.layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.final_layer = nn.Linear(layers[-2], layers[-1])
        
        # U1, b1, U2, b2
        self.U1 = nn.Parameter(torch.zeros(layers[0], layers[1]))
        self.b1 = nn.Parameter(torch.zeros(layers[1]))
        self.U2 = nn.Parameter(torch.zeros(layers[0], layers[1]))
        self.b2 = nn.Parameter(torch.zeros(layers[1]))

    def forward(self, x):
        U = self.activation(x @ self.U1 + self.b1)
        V = self.activation(x @ self.U2 + self.b2)
        h = x
        for layer in self.layers:
            out = self.activation(layer(h))
            h = out * U + (1 - out) * V
        return self.final_layer(h)

# -------------------------
# PI_DeepONet
# -------------------------
class TorchPI_DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super().__init__()
        
        # self.branch_net = TorchModifiedMLP(branch_layers)
        # self.trunk_net = TorchModifiedMLP(trunk_layers)
        self.branch_net = TorchMLP(branch_layers)
        self.trunk_net = TorchMLP(trunk_layers)
    
    def forward(self, u, y):
        B = self.branch_net(u)
        T = self.trunk_net(y)
        return torch.sum(B * T, dim=-1, keepdim=True)

# -------------------------
# JAX -> PyTorch 
# -------------------------
def load_modifiedMLP_jax_to_torch(jax_params, torch_model):
    params, U1, b1, U2, b2 = jax_params
    
    # U1, b1, U2, b2
    torch_model.U1.data = torch.tensor(np.array(U1), dtype=torch.float32)
    torch_model.b1.data = torch.tensor(np.array(b1), dtype=torch.float32)
    torch_model.U2.data = torch.tensor(np.array(U2), dtype=torch.float32)
    torch_model.b2.data = torch.tensor(np.array(b2), dtype=torch.float32)
    
    #  W, b
    for i, (W, b) in enumerate(params[:-1]):
        torch_model.layers[i].weight.data = torch.tensor(np.array(W.T), dtype=torch.float32)
        torch_model.layers[i].bias.data = torch.tensor(np.array(b), dtype=torch.float32)
    
    W, b = params[-1]
    torch_model.final_layer.weight.data = torch.tensor(np.array(W.T), dtype=torch.float32)
    torch_model.final_layer.bias.data = torch.tensor(np.array(b), dtype=torch.float32)
    
def load_MLP_jax_to_torch(jax_params, torch_model):
    
    for i, (W, b) in enumerate(jax_params):
        layer = torch_model.layers[i]
        # 转置 W，因为 JAX 是 (in,out)，PyTorch 是 (out,in)
        layer.weight.data = torch.tensor(np.array(W.T), dtype=torch.float32)
        layer.bias.data = torch.tensor(np.array(b), dtype=torch.float32)


if __name__ == "__main__":
    # 加载 JAX 模型参数
    jax_params_flat = np.load('TrainedModels/model_jax_iter40000.npy')
    m = 100
    branch_layers = [m, 50, 50, 50, 50, 50]
    trunk_layers =  [1, 50, 50, 50, 50, 50]
    
    jax_model = PI_DeepONet(branch_layers, trunk_layers)
    jax_params = jax_model.unravel_params(jax_params_flat)
    branch_params, trunk_params = jax_params


    torch_model = TorchPI_DeepONet(branch_layers, trunk_layers)
    torch_model.eval()


    # load_modifiedMLP_jax_to_torch(branch_params, torch_model.branch_net)
    # load_modifiedMLP_jax_to_torch(trunk_params, torch_model.trunk_net)
    
    load_MLP_jax_to_torch(branch_params, torch_model.branch_net)
    load_MLP_jax_to_torch(trunk_params, torch_model.trunk_net)


    key = random.PRNGKey(0)
    u_test = random.uniform(key, (10, branch_layers[0]))
    y_test = random.uniform(key, (10, trunk_layers[0]))


    jax_out = vmap(jax_model.operator_net, (None, 0, 0))(jax_params, u_test, y_test)


    u_torch = torch.tensor(np.array(u_test), dtype=torch.float32)
    y_torch = torch.tensor(np.array(y_test), dtype=torch.float32)
    torch_out = torch_model(u_torch, y_torch).detach().numpy()

    print("最大差异:", np.max(np.abs(jax_out.reshape(-1,1) - torch_out)))
    print("均方差:", np.mean((jax_out.reshape(-1,1) - torch_out)**2))
    
    torch.save(torch_model, 'TrainedModels/torch_model.pth')
