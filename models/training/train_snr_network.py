from torch import nn
import torch
import numpy as np
from mlsel.nn.model_train_test import model_train_test
from mlsel.nn.model_creation import create_mlp
import pandas as pd

if __name__ == '__main__':
    torch.set_num_threads(1)
    device = "cuda:1"
    fp = '/home/christian/emri_selection_paper/snr_datasets/T10/{}'
    outdir = './models/snr/'
    train_inds = [0,1,2,3,4,5,6,7,8] #['logM', 'logq', 'a', 'e0', 'Y0', 'thetaS', 'phiS-phiK', 'thetaK', 'tplunge', 'SNR']
    test_inds = [9,]

    traindata = pd.read_csv(fp.format('4yr.csv'))
    xtrain = traindata.iloc[:,train_inds].to_numpy()
    ytrain = traindata.iloc[:,test_inds].to_numpy()

    main_cut = 1.
    xtrain = xtrain[:int(main_cut*xtrain.shape[0]),:]
    ytrain = ytrain[:int(main_cut*ytrain.shape[0]),:]

    inds = np.arange(ytrain.shape[0])
    np.random.shuffle(inds)
    xtrain = xtrain[inds,:]
    ytrain = ytrain[inds,:]

    keep = np.where(ytrain != 0)[0]
    xtrain = xtrain[keep, :]
    # ytrain = np.log(ytrain[keep, :])
    ytrain = ytrain[keep, :]

    cut = 0.9
    xtest = xtrain[int(cut*xtrain.shape[0]):,:]
    ytest = ytrain[int(cut*ytrain.shape[0]):,:]
    xtrain = xtrain[:int(cut*xtrain.shape[0]),:]
    ytrain = ytrain[:int(cut*ytrain.shape[0]),:]

    print('Train, test |', ytrain.shape[0], ',', ytest.shape[0])
    
    in_features = len(train_inds)
    out_features = 1
    layers = 10
    n_batches = 5
    neurons = np.array(np.ones(layers,dtype=np.int32)*128).tolist()
    activation = nn.SiLU
    out_activation=None
    model = create_mlp(input_features=in_features,output_features=out_features,neurons=neurons,layers=layers,activation=activation,out_activation=out_activation, 
    device=device, model_name=f'T10_LR1e-4_{layers}L_{n_batches}B_{neurons[0]}N', outdir=outdir)

    data = [xtrain, ytrain, xtest, ytest]

    loss_function = nn.L1Loss()
    LR = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sched = None#torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=5e-5,max_lr=1e-3, step_size_up=10000,mode='triangular2', cycle_momentum=False)

    model_train_test(data, model, device,n_epochs=100000,n_batches=n_batches, loss_function=loss_function,optimizer=optimizer, verbose=True,update_every=1000, n_test_batches=1,save_best=True, scheduler=sched,  outdir=outdir)
