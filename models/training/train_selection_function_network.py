from torch import nn
import torch
import numpy as np
from utils import model_train_test, create_mlp
import pandas as pd

if __name__ == '__main__':
    torch.set_num_threads(1)
    device = "cuda:1"
    train_inds = [0,1,2,3,4,5,6,7]
    test_inds = [8,]
    traindata = pd.read_csv('/home/christian/emri_selection_paper/selection_function_datasets/T2_linear_1e5_1e5_2.csv')
    outdir = './models/selection/'
    xtrain = traindata.iloc[:,train_inds].to_numpy()
    ytrain = traindata.iloc[:,test_inds].to_numpy()

    inds = np.arange(ytrain.shape[0])
    np.random.shuffle(inds)
    xtrain = xtrain[inds,:]
    ytrain = np.log(ytrain[inds,:])
    
    cut = 0.99
    xtest = xtrain[int(cut*xtrain.shape[0]):,:]
    ytest = ytrain[int(cut*ytrain.shape[0]):,:]
    xtrain = xtrain[:int(cut*xtrain.shape[0]),:]
    ytrain = ytrain[:int(cut*ytrain.shape[0]),:]
    
    print('Train, test |', ytrain.shape[0], ',', ytest.shape[0])

    in_features = len(train_inds)
    out_features = len(test_inds)
    layers = 8
    n_batches = 5
    neurons = np.array(np.ones(layers,dtype=np.int32)*128).tolist()
    activation = nn.SiLU
    out_activation=None 
    model = create_mlp(input_features=in_features,output_features=out_features,neurons=neurons,layers=layers,
    activation=activation,out_activation=out_activation, device=device, model_name=f'T2_linear_{layers}L_LR1e-4_{n_batches}B_{neurons[0]}N_log',outdir=outdir)

    data = [xtrain, ytrain, xtest, ytest]

    loss_function = nn.L1Loss()
    LR = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sched = None#torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=5e-5,max_lr=1e-3, step_size_up=10000,mode='triangular2', cycle_momentum=False)
    
    model_train_test(data, model, device,n_epochs=100000,n_batches=n_batches, loss_function=loss_function,optimizer=optimizer, verbose=True,update_every=500, n_test_batches=1,save_best=True, scheduler=sched, outdir=outdir)
