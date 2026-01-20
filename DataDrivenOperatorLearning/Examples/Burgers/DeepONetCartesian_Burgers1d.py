import sys 
sys.path.append("../..") 
import numpy as np
import h5py
import torch 
#
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
random_seed = 1234
setup_seed(random_seed)
device = 'cuda:0'
dtype = torch.float32
problem_name = 'Burgers_1d'
######################################
# Load training data
######################################
data_train = h5py.File('../Data/Burgers_1d/viscid_train.mat', 'r')
data_test = h5py.File('../Data/Burgers_1d/viscid_test_in.mat', 'r')
print(data_train.keys())
print(data_test.keys())
######################################
# Load training data
######################################
from Utils.utils import *
n_train, n_test = 1000, 50
def get_data(data, ndata, dtype, n0=0):
    # Data is of the shape (number of samples = 1000, grid size = 29*29)
    a = np2tensor(np.array(data["u0"][...,n0:n0+ndata]).T, dtype)
    u = np2tensor(np.array(data["u_sol"][...,n0:n0+ndata]).T, dtype)
    uT = u[:,-1,:]
    #
    x_mesh = np2tensor(np.array(data['x_mesh']))
    #
    a = a.reshape(ndata, -1)
    uT = uT.reshape(ndata, -1, 1)
    
    return a, uT, x_mesh
#
a_train, uT_train, gridx_train = get_data(data_train, n_train, dtype)
a_test, uT_test, gridx_test = get_data(data_test, n_test, dtype)
#
print('The shape of a_train:', a_train.shape)
print('The shape of uT_train:', uT_train.shape)
print('The shape of gridx_train:', gridx_train.shape)
#
print('The shape of a_test:', a_test.shape)
print('The shape of uT_train:', uT_test.shape)
print('The shape of gridx_test:', gridx_test.shape)
########################################
from Utils.PlotFigure import Plot
inx = 5
#
Plot.show_1d_list(gridx_train, [a_train[inx], uT_train[inx]], ['a0_train', 'uT0_train'], lb =-1.)

###############################
# Set normalizer
###############################
from sklearn.preprocessing import StandardScaler

class Normalizer_a(object):

    def __init__(self, scale:float=0.1, shift:float=0.75):
        self.scale = scale
        self.shift = shift
        
    def encode(self, a:torch.tensor):
        '''
        Input: 
            a: (n_batch, n_mesh) 
        '''
        return a * self.scale - self.shift
        
    def decode(self, a:torch.tensor):
        '''
        Input: 
            a: (n_batch, n_mesh)
        '''
        return (a+self.shift)*self.scale
#
normalizer_a = Normalizer_a(1., 0.)

###############################
class LossClass(object):

    def __init__(self, solver):
        super(LossClass, self).__init__()
        ''' '''
        self.solver = solver
        self.dtype = solver.dtype
        self.device = solver.device
        self.model_u = solver.model_dict['u']

    def Loss_data(self, a, u, x_grid):
        ''' '''
        a_norm = normalizer_a.encode(a)
        u_pred = self.model_u(x_grid, a_norm)
        #
        loss = self.solver.getLoss(u_pred, u)
        
        return loss

    def Error(self, a, u, x_grid):
        ''' '''
        a_norm = normalizer_a.encode(a)
        u_pred = self.model_u(x_grid, a_norm)
        #
        err = self.solver.getError(u_pred, u)
        
        return err

######################################
# Steups of the model
######################################
from Solvers.DeepONet import DeepONet
solver = DeepONet.Solver(device, dtype)
netType = 'DeepONetCartesian_Tanh_Sin'
#
layers_branch, activation_branch = [128, 128, 128, 128, 128], 'Tanh_Sin'
layers_trunk, activation_trunk = [1, 128, 128, 128, 128], 'Tanh_Sin'
model_u = solver.getModel(layers_branch, layers_trunk, activation_branch, activation_trunk, 
                        multi_ouput_strategy=None, num_output=1, netType='Cartesian')
# # ###############################
from torchsummary import summary
summary(model_u, [(1,), (128,)], device='cuda')
# ###############################
total_params = sum(p.numel() for p in model_u.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params_u = sum(p.numel() for p in model_u.parameters() if p.requires_grad)
print(f'{total_trainable_params_u:,} training parameters.')


model_dict = {'u':model_u}
solver.train_setup(model_dict, lr=1e-3, optimizer='Adam', scheduler_type='StepLR', step_size=500)
solver.train_cartesian(LossClass, a_train, uT_train, gridx_train, a_test, uT_test, gridx_test, 
                       batch_size=50, epochs=2000, epoch_show=100, **{'save_path':f'saved_models/{netType}/'})

#####################################
# Load the trained model
#####################################
from Solvers.DeepONet import DeepONet
solver = DeepONet.Solver(device, dtype)
model_trained = solver.loadModel(path=f'saved_models/{netType}/', name='model_deeponet_final')

#########################################
with torch.no_grad():
    a = normalizer_a.encode(a_test.to(device))
    uT_pred = model_trained['u'](gridx_test.to(device), a)
    uT_pred = uT_pred.detach().cpu()
#
print('The shape of a_test:', a_test.shape)
print('The shape of uT_test:', uT_test.shape)
print('The test loss', solver.getLoss(uT_pred, uT_test))
print('The test l2 error:', solver.getError(uT_pred, uT_test))
print('*************************************')
for i in range(0, 5):
    print(f'The test l2 error for {i}:', solver.getError(uT_pred[i:i+1], uT_test[i:i+1]))
inx = 0
# ########################################
from Utils.PlotFigure import Plot
# show prediction
Plot.show_1d_list(gridx_test, [uT_test[inx], uT_pred[inx]], 
                  label_list=['uT0_True', 'uT0_Pred'], title='prediction of uT')
# show loss
loss_saved = solver.loadLoss(path=f'saved_models/{netType}/', name='loss_deeponet')
Plot.show_loss([loss_saved['loss_train'], loss_saved['loss_test']], ['loss_train', 'loss_test'])
# show error
Plot.show_error([loss_saved['time']], [loss_saved['l2_test']], ['l2_test'])


