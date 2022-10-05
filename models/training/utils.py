import torch
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
from pathlib import Path
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import dill as pickle
import os

def get_script_path():
    return os.getcwd()

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, neurons, n_layers, activation, name, out_activation=None, initialisation=xavier_uniform_, dropout=False, batch_norm=False):
        super().__init__()
        self.initial = initialisation
        self.name = name

        if isinstance(neurons, int):
            neurons = np.ones_like(n_layers)*neurons
        elif isinstance(neurons, list):
            if len(neurons) != n_layers:
                raise Exception("number of neurons and number of layers do not agree")

        layers = [nn.Linear(in_features, neurons[0]), activation()]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            if dropout is not False:
                layers.append(nn.Dropout(dropout))  
            layers.append(activation())
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=neurons[i+1]))
                
        layers.append(nn.Linear(neurons[-1], out_features))
        if out_activation is not None:
            layers.append(out_activation())
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.initial(m.weight)


def create_mlp(input_features, output_features, neurons, layers, activation, model_name, out_activation=None, init=xavier_uniform_, device=None, norm_type='z-score', dropout=False, batch_norm=False, outdir='../models'):
    if isinstance(neurons, list):
        if len(neurons) != layers:
            raise RuntimeError('Length of neuron vector does not equal number of hidden layers.')
    else:
        neurons = [neurons, ]
    model = LinearModel(input_features, output_features, neurons, layers, activation, model_name, initialisation=init, dropout=dropout, batch_norm=batch_norm, out_activation=out_activation)
    model.norm_type=norm_type
    Path(get_script_path()+f'/{outdir}/{model_name}/').mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(get_script_path()+f'/{outdir}/{model_name}/function.pickle', "wb"), pickle.HIGHEST_PROTOCOL)  # save blank model

    if device is not None:
        model = model.to(device)
    return model


def model_train_test(data, model, device, n_epochs, n_batches, loss_function, optimizer, verbose=False, return_losses=False, update_every=None, n_test_batches=None, save_best=False, scheduler=None, outdir='../models'):
    
    if n_test_batches is None:
        n_test_batches = n_batches
    
    xtrain, ytrain, xtest, ytest = data
    model.to(device)

    name = model.name
    path = get_script_path()
    norm_type = model.norm_type
    Path(get_script_path()+f'/{outdir}/{name}/').mkdir(parents=True, exist_ok=True)
    Path(get_script_path()+f'/{outdir}/{name}/comparisons').mkdir(parents=True, exist_ok=True)
    if norm_type == 'z-score':
        np.save(path+f'/{outdir}/{name}/xdata_inputs.npy',np.array([xtrain.mean(axis=0), xtrain.std(axis=0)]))
        np.save(path+f'/{outdir}/{name}/ydata_inputs.npy',np.array([ytrain.mean(), ytrain.std()]))
    elif norm_type == 'uniform':
        np.save(path+f'/{outdir}/{name}/xdata_inputs.npy',np.array([np.min(xtrain,axis=0), np.max(xtrain,axis=0)]))
        np.save(path+f'/{outdir}/{name}/ydata_inputs.npy',np.array([np.min(ytrain), np.max(ytrain)]))
    elif norm_type is None:
        pass
    xtest = torch.from_numpy(norm_inputs(xtest, ref_dataframe=xtrain, norm_type=norm_type)).to(device).float()
    ytest = torch.from_numpy(norm(ytest, ref_dataframe=ytrain, norm_type=norm_type)).to(device).float()
    xtrain = torch.from_numpy(norm_inputs(xtrain, ref_dataframe=xtrain, norm_type=norm_type)).to(device).float()
    ytrain = torch.from_numpy(norm(ytrain, ref_dataframe=ytrain, norm_type=norm_type)).to(device).float()

    ytrainsize = len(ytrain)
    ytestsize = len(ytest)

    train_losses = []
    test_losses = []
    rate = []
    # Run the training loop

    datasets = {"train": [xtrain, ytrain], "test": [xtest, ytest]}

    cutoff_LR = n_epochs - 50
    lowest_loss = 1e5
    for epoch in range(n_epochs):  # 5 epochs at maximum
        # Print epoch
        for phase in ['train','test']:
            if phase == 'train':
                model.train(True)
                shuffled_inds = torch.randperm(ytrainsize)

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                # Get and prepare inputs
                inputs, targets = datasets[phase]
                inputs = inputs[shuffled_inds]
                targets = targets[shuffled_inds]

                #targets = targets.reshape((targets.shape[0], 1))

                for i in range(n_batches):
                    for param in model.parameters():
                        param.grad = None
                    outputs = model(inputs[i * ytrainsize // n_batches:(i+1)*ytrainsize // n_batches])
                    loss = loss_function(outputs, targets[i * ytrainsize // n_batches: (i+1)*ytrainsize // n_batches])
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    current_loss += loss.item()

                train_losses.append(current_loss / n_batches)

            else:
                with torch.no_grad():
                    model.train(False)
                    shuffled_inds = torch.randperm(ytestsize)
                    current_loss = 0.0
                    inputs, targets = datasets[phase]
                    inputs = inputs[shuffled_inds]
                    targets = targets[shuffled_inds]

#                     targets = targets.reshape((targets.shape[0], 1))

                    for i in range(n_test_batches):
                        outputs = model(inputs[i * ytestsize // n_test_batches: (i+1)*ytestsize // n_test_batches])
                        loss = loss_function(outputs, targets[i * ytestsize // n_test_batches: (i+1)*ytestsize // n_test_batches])
                        current_loss += loss.item()

                    test_losses.append(current_loss / n_test_batches)
        if test_losses[-1] < lowest_loss:
            lowest_loss = test_losses[-1]
            if save_best:
                torch.save(model.state_dict(),path+f'/{outdir}/{name}/model.pth')
                
#         if epoch >= cutoff_LR:
#             scheduler.step()
#             rate.append(scheduler.get_last_lr()[0])
#         else:
#             rate.append(learning_rate)
        if verbose:
            stdout.write(f'\rEpoch: {epoch} | Train loss: {train_losses[-1]:.3e} | Test loss: {test_losses[-1]:.3e} (Lowest: {lowest_loss:.3e})')
        if update_every is not None:
            if not epoch % update_every:# and epoch != 0:
                epochs = np.arange(epoch+1)
                plt.semilogy(epochs, train_losses, label='Train')
                plt.semilogy(epochs, test_losses, label='Test')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Train and Test Loss Across Train Epochs')
                plt.savefig(path+f'/{outdir}/{name}/losses.png')
                #plt.show()
                plt.close()
                
                comparison_plot(targets, outputs, path+f'/{outdir}/{name}/comparisons/{epoch}.png')

                if not save_best:
                    torch.save(model.state_dict(),path+f'/{outdir}/{name}/model.pth')

        
    if verbose:
        print('\nTraining complete - saving.')
    
    if not save_best:
        torch.save(model.state_dict(),path+f'/{outdir}/{name}/model.pth')

    epochs = np.arange(n_epochs)
    plt.semilogy(epochs, train_losses, label='Train')
    plt.semilogy(epochs, test_losses, label='Test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Across Train Epochs')
    plt.savefig(path+f'/{outdir}/{name}/losses.png')
    #plt.show()
    plt.close()

    out = (model,)
    if return_losses:
        out += (train_losses, test_losses,)
    return out

def comparison_plot(true, pred, save_path=None):
    try:
        true = true.cpu()
        pred = pred.cpu()
    except:
        pass
    plt.figure()
    plt.scatter(true.numpy(), pred.numpy(), s=1, label='Network outputs')
    plt.plot([true.min(),true.max()],[true.min(),true.max()], label='1:1', c='k')
    plt.legend()
    plt.xlabel('Labels (normalised)')
    plt.ylabel('Predictions (normalised)')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()    


def norm(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if norm_type is None:
        return dataframe
    elif ref_dataframe is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_dataframe.mean())/ref_dataframe.std()
        elif norm_type == 'uniform':
            df_norm = 2*((dataframe - ref_dataframe.min())/ref_dataframe.max()) - 1
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_inputs[0])/ref_inputs[1]
        elif norm_type == 'uniform':
            df_norm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
        else:
            raise Exception("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_norm


def norm_inputs(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if norm_type is None:
        return dataframe
    elif ref_dataframe is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_dataframe.mean(axis=0))/ref_dataframe.std(axis=0)
        elif norm_type == 'uniform':
            df_norm = 2*((dataframe - ref_dataframe.min(axis=0))/ref_dataframe.max(axis=0)) - 1
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_inputs[0])/ref_inputs[1]
        elif norm_type == 'uniform':
            df_norm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
        else:
            raise Exception("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_norm

def unnorm(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if norm_type is None:
        return dataframe
    elif ref_dataframe is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_dataframe.std()) + ref_dataframe.mean()
        elif norm_type == 'uniform':
            #df_unnorm = 2*((dataframe - np.min(ref_dataframe))/np.max(ref_dataframe)) - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_dataframe.max() + ref_dataframe.min()
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_inputs[1]) + ref_inputs[0]
        elif norm_type == 'uniform':
            #df_unnorm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_inputs[1] + ref_inputs[0]
        else:
            raise Exception("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_unnorm

def unnorm_inputs(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if norm_type is None:
        return dataframe
    elif ref_dataframe is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_dataframe.std(axis=0)) + ref_dataframe.mean(axis=0)
        elif norm_type == 'uniform':
            #df_unnorm = 2*((dataframe - np.min(ref_dataframe))/np.max(ref_dataframe)) - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_dataframe.max(axis=0) + ref_dataframe.min(axis=0)
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_inputs[1]) + ref_inputs[0]
        elif norm_type == 'uniform':
            #df_unnorm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_inputs[1] + ref_inputs[0]
        else:
            raise Exception("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_unnorm

