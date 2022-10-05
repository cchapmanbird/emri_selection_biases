import numpy as np
import pandas as pd
from model import dVdz_over_opz, truncnorm, powerlaw, dL
from scipy.interpolate import CubicSpline
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np
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

def construct_z_sample_generator(z_max, nz=1000):
    z_samples = np.linspace(0,z_max,nz)
    p_z_vals = np.asarray([dVdz_over_opz(z) for z in z_samples])
    cdf = np.cumsum(p_z_vals)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0]) # this is fine here due to uniform spacing
    spline = CubicSpline(cdf, z_samples)
    return spline

def construct_dL_lookup(z_max, nz=1000):
    z_in = np.linspace(0, z_max, nz)
    dL_out = np.asarray([dL(z) for z in z_in])
    
    spline = CubicSpline(z_in, dL_out)
    return spline

def construct_dVdz_spline(z_max, nz=1000):
    z_samples = np.linspace(0,z_max,nz)
    p_z_vals = np.asarray([dVdz_over_opz(z) for z in z_samples])
    spline = CubicSpline(z_samples, p_z_vals)
    return spline

def construct_cosmo_splines(z_max, nz=1000):
    spl1 = construct_z_sample_generator(z_max, nz=nz)
    spl2 = construct_dL_lookup(z_max, nz=nz)
    spl3 = construct_dVdz_spline(z_max, nz=nz)
    return spl1, spl2, spl3

def ensure_arrays(*params):
    converted_params = [xp.asarray(param) for param in params]
    return converted_params


class Distribution(object):
    def __init__(self, pdf, use_gpu=False):
        self.pdf = pdf
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
            
    def draw_samples(self, min, max, size=1, pdf_args=[], pdf_kwargs={}, scaling='linear', grid_size=100):
        if scaling == 'linear':
            param_grid = self.xp.linspace(min, max, grid_size)
        elif scaling == 'log':
            param_grid = self.xp.logspace(self.xp.log(min),self.xp.log(max), grid_size)
        grid_spacing = self.xp.append(0,self.xp.diff(param_grid))
        renorm = grid_spacing * probs
        probs = self.pdf(param_grid, *pdf_args, **pdf_kwargs)
        probs *= 1/renorm.sum()
        cdf = self.xp.cumsum(renorm)
        draws = self.xp.interp(self.xp.random.random(size), cdf, param_grid)
        return draws

def draw_samples_from_population(z_generator, dL_lookup, max_T, grid_size, size, return_as, lambda_M, Mmin, Mmax, lambda_mu, mumin, mumax, mu_a, sigma_a):

    lambda_M, Mmin, Mmax, lambda_mu, mumin, mumax, mu_a, sigma_a = ensure_arrays(lambda_M, Mmin, Mmax, lambda_mu, mumin, mumax, mu_a, sigma_a)
    samples = xp.zeros((size, 10))

    #a
    amin = max(0.001, mu_a - 5*sigma_a)
    amax = min(0.999, mu_a + 5*sigma_a)
    a_values = xp.linspace(amin, amax, grid_size)
    apdf = truncnorm(a_values, mu_a, sigma_a, xp.asarray(0.999), xp.asarray(0.001))
    apdf *= 1/(apdf * xp.append(0,xp.diff(a_values))).sum()
    acdf = xp.cumsum(apdf * xp.append(0,xp.diff(a_values)))
    a_draws = xp.interp(xp.random.random(size), acdf, a_values)
    samples[:,2] = a_draws
    #Y0
    samples[:,4] = xp.random.uniform(0.5, 0.99, size) * (2*xp.random.randint(2, size=int(size))-1)
    # thetaS, phiS-phiK, thetaK
    samples[:,6] = xp.arccos(2*xp.random.uniform(0,1,size=size)-1)
    samples[:,7] = xp.random.uniform(0,2*xp.pi,size=size)
    samples[:,8] = xp.arccos(2*xp.random.uniform(0,1,size=size)-1)

    # Tplunge
    samples[:,9] = xp.random.uniform(0,max_T,size=size)
    
    # z, dL
    z_draws = z_generator(np.random.random(size))
    samples[:,5] = xp.asarray(dL_lookup(z_draws))

    # mu
    mu_values = xp.logspace(xp.log10(mumin),xp.log10(mumax),grid_size)
    mupdf = powerlaw(mu_values, lambda_mu, mumin, mumax)
    mupdf *= 1/(mupdf * xp.append(0,xp.diff(mu_values))).sum()
    mucdf = xp.cumsum(mupdf * xp.append(0,xp.diff(mu_values)))
    mu_draws = xp.interp(xp.random.random(size), mucdf, mu_values)

    M_values = xp.logspace(xp.log10(Mmin),xp.log10(Mmax),grid_size)
    mpdf = powerlaw(M_values, lambda_M, Mmin, Mmax)
    mpdf *= 1/(mpdf * xp.append(0,xp.diff(M_values))).sum()
    mcdf = xp.cumsum(mpdf * xp.append(0,xp.diff(M_values)))
    M_draws = xp.interp(xp.random.random(size), mcdf, M_values)

    z_draws = xp.asarray(z_draws)
    # convert M, mu to log(M(1+z)), logq
    samples[:,0] = xp.log10(M_draws*(1+z_draws))
    samples[:,1] = xp.log10(mu_draws / M_draws)

    # e0
    samples[:,3] = xp.random.uniform(0.1, 0.5, size)

    # output
    if return_as == 'cupy':
        out = samples
    else:
        try:
            samples = samples.get()
        except AttributeError:
            pass
        out = pd.DataFrame(samples,columns=['logM','logq','a','e0','Y0','dL','thetaS','phiS-phiK','thetaK', 'tplunge'])
    return out
