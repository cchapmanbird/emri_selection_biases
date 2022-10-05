import numpy as np
try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates
except:
    xp = np
    from scipy.ndimage import map_coordinates
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import os
import dill as pickle

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

def load_mlp(model_name, device, get_state_dict=False, outdir='../models'):
    model = pickle.load(open(get_script_path()+f'/{outdir}/{model_name}/function.pickle', "rb"))  # load blank model
    if get_state_dict:
        model.load_state_dict(torch.load(open(get_script_path()+f'/{outdir}/{model_name}/model.pth', "rb"), map_location=device))
    if model.norm_type is not None:
        xscalevals = torch.as_tensor(np.load(get_script_path() + f'/{outdir}/{model.name}/xdata_inputs.npy'), device=device).float()
        yscalevals = torch.as_tensor(np.load(get_script_path() + f'/{outdir}/{model.name}/ydata_inputs.npy'), device=device).float()
    else:
        xscalevals = None
        yscalevals = None
    model.xscalevals = xscalevals
    model.yscalevals = yscalevals
    return model


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


def run_on_dataset(model, xdata, distances=None, n_batches=1, device=None, y_transform_fn=None, runtime=False,
                    eval_model = True):
    """
    Get the re-processed output of the supplied model on a set of supplied test data.

    Args:
        model (LinearModel): Model to test on `xdata`
        xdata (ndarray): Array of features to test against
        distances (ndarray): List of luminosity distance measurements for the input events. If None, results will not be scaled by luminosity distance. Note that this scaling is applied after the data is converted with y_transform_fn.
        n_batches (int, optional): Number of batches to process the input data in. Defaults to 1 (the entire dataset).
        device (string, optional): Device to attach the input model to, if it is not attached already.
        y_transform_fn (function, optional): If the labels/ydata have been pre-processed with a function (e.g. log),
                                             this function must be supplied to undo this if comparison is to be made
                                             in the unaltered function space.
        runtime (bool, optional): If True, return timing statistics for the model on the provided dataset.
                                  Defaults to False.

    Returns:
        tuple containing:
            output_data (double): Neural network output for this dataset.
            total_time (double): total time taken by the model to process the input dataset.
                                 Only if `runtime` is set to True.
            per_point (double): mean time taken by the model per sample in the input dataset.
                                 Only if `runtime` is set to True.
    """
    if device is not None:
        model = model.to(device)
    if eval_model:
        model.eval()

    test_input = torch.as_tensor(xdata, device=device).float()
    normed_input = norm_inputs(test_input, ref_inputs=model.xscalevals,norm_type=model.norm_type)

    if n_batches > 1:
        with torch.no_grad():
            out = []
            for _ in range(n_batches):
                output = model(normed_input)
                out.append(output)
            output = torch.cat(out)
    else:
        with torch.no_grad():
            output = model(normed_input)

    try:
        if output.shape[1] == 1:
            output = output[:,0]        
    except IndexError:
        pass

    out_unnorm = unnorm(output, ref_inputs=model.yscalevals,norm_type=model.norm_type)

    if y_transform_fn is not None:
        out_unnorm = y_transform_fn(out_unnorm)
    
    if distances is not None:
        out_unnorm *= (1/distances)#[:,None]

    outputs = out_unnorm
    return outputs


def cumulative_dist(vals):
    sortedv = np.sort(vals)
    cdf = np.linspace(0, 1, len(sortedv), endpoint=False)
    return sortedv, cdf

def model_wrapper(model, param_inds, samples):
    xdata = samples[:,param_inds]
    out = model(xdata)
    try:
        out = out.get()
    except AttributeError:
        pass
    return out

def nn_model_wrapper(model, device, param_inds, samples):
    samples[:,1] = xp.log10(samples[:,1]) - samples[:,0]
    xdata = samples[:,param_inds]
    out = run_on_dataset(model,xdata,distances=None,
                               device=device,y_transform_fn=None,runtime=False, eval_model=False)
    try:
        out = out.get()
    except AttributeError:
        pass
    return out

def rescale_to_limits(samples, limits, grid_dims):
    return (samples - limits[:,0]) / (limits[:,1] - limits[:,0]) * (grid_dims - 1)

def rescale_and_map(grid, limits, grid_dimensions, order, samples):
    ready = rescale_to_limits(samples, limits, grid_dimensions).T
    outs = map_coordinates(grid, ready, order=order, cval=-10,mode='constant')
    return outs

