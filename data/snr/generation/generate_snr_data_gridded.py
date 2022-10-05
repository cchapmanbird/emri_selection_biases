import numpy as np
from utils import window_waveform, vectorised_snr2, cornish_lisa_psd
import pandas as pd
from pathlib import Path
from few.waveform import GenerateEMRIWaveform, EMRIInspiral
from few.utils.utility import get_p_at_t, get_separatrix, Y_to_xI, omp_set_num_threads
import cupy as xp
from few.utils.constants import YRSID_SI
omp_set_num_threads(1)
aak = GenerateEMRIWaveform("Pn5AAKWaveform", inspiral_kwargs=dict(max_init_len=int(1e7)),sum_kwargs=dict(pad_output=True), use_gpu=True)
traj = EMRIInspiral(func="pn5")
traj_inds = [0,1,2,4,5,11,12,13]

outdir = '/home/christian/emri_selection_paper/snr_datasets/T10/{}'
Path(outdir[:-3]).mkdir(parents=True, exist_ok=True)

T = 2.
dt = 10.

# run waveform once to cache things
injection_parameters = np.zeros(14)
injection_parameters[0] = 1e6
injection_parameters[1] = 1e0
injection_parameters[2] = 0.5
injection_parameters[3] = 100.
injection_parameters[4] = 0.2
injection_parameters[5] = 0.9
injection_parameters[6] = 1.
wave = aak(*injection_parameters, dt=dt, T=T)
shortened_wave = window_waveform(wave, T, dt, 2., window_size=2.)
nwavesamps = shortened_wave.size
fs = xp.fft.rfftfreq(shortened_wave.size, d=dt)[1:]
dfs = xp.zeros_like(fs)
dfs[1:] = xp.diff(fs)
dfs[0] = dfs[1]
psd_values = cornish_lisa_psd(fs, duration=2)

grid_dims = [5,5,3,3,3,5,7,5,10]
every = np.prod(grid_dims[-4:-1])
overall_total = np.prod(grid_dims)
print('Number in the grid:', overall_total, '| Waveforms:', overall_total // grid_dims[-1])
cols = ['logM', 'mu', 'a', 'e0', 'Y0', 'thetaS', 'phiS-phiK', 'thetaK','tplunge']

lMs = np.linspace(np.log10(5e4),8,grid_dims[0])
mus = np.linspace(1,200,grid_dims[1])
avs = np.linspace(0.001, 0.999, grid_dims[2])
es = np.linspace(0.1, 0.5, grid_dims[3])
ys = np.linspace(0.5, 0.99, grid_dims[4])
qSs = np.linspace(0, np.pi, grid_dims[5])
tSs= np.linspace(0,2*np.pi,grid_dims[6])
qKs = np.linspace(0, np.pi, grid_dims[7])
tpls = np.linspace(0, 2, grid_dims[8])

mesh = np.meshgrid(lMs, mus, avs, es, ys, qSs, tSs, qKs, tpls, indexing='ij')
mesh2 = np.meshgrid(lMs, mus, avs, es, ys, qSs, tSs, qKs, indexing='ij')

inp = [np.ravel(meshparam) for meshparam in mesh]
all_event_params = np.vstack(inp).T
inp2 = [np.ravel(meshparam) for meshparam in mesh2]
event_params = np.vstack(inp2).T

#df = pd.DataFrame(all_event_params, columns=cols)
#snrs_out = xp.zeros(overall_total)
df = pd.read_csv('/home/christian/emri_selection_paper/snr_datasets/T10/4yr_grid5.csv')
snrs_out = xp.asarray(df['SNR'].to_numpy())

#i = 113001#0
for k, paramset in enumerate(event_params[112875:,:], start=112875):
    waveforms = xp.zeros((grid_dims[-1],nwavesamps), dtype=xp.complex64)
    injection_parameters[0] = 10**paramset[0]
    injection_parameters[1] = paramset[1]
    injection_parameters[2] = paramset[2]
    injection_parameters[4:6] = paramset[3:5]
    injection_parameters[7:10] = paramset[5:8]
    # if k == 113050:
    #     breakpoint()
    if not k % every:
        sep_guess1 = get_separatrix(injection_parameters[2],injection_parameters[4],injection_parameters[5])
        xI = Y_to_xI(injection_parameters[2], sep_guess1,injection_parameters[4],injection_parameters[5])
        sep_guess2 = get_separatrix(injection_parameters[2],injection_parameters[4],xI)
        xI = Y_to_xI(injection_parameters[2], sep_guess2,injection_parameters[4],injection_parameters[5])
        sep_guess2 = get_separatrix(injection_parameters[2],injection_parameters[4],xI)
        try:
            injection_parameters[3] = get_p_at_t(traj, T, np.take(injection_parameters,traj_inds).tolist(),
                                    bounds = [sep_guess2+0.100001, 100.], traj_kwargs=dict(max_init_len=int(1e7)))
        except:
            if injection_parameters[0] > 1e6:
                injection_parameters[3] = sep_guess2 + 0.5
            else:
                try:
                    injection_parameters[3] = get_p_at_t(traj, T, np.take(injection_parameters,traj_inds).tolist(),
                                        bounds = [100., 200.], traj_kwargs=dict(max_init_len=int(1e7)))
                except:
                    injection_parameters[3] = get_p_at_t(traj, T, np.take(injection_parameters,traj_inds).tolist(),
                        bounds = [200., 500.], traj_kwargs=dict(max_init_len=int(1e7)))

    print(f'Event {k}, {injection_parameters[:10]}')
    while True:
        try:
            wave = aak(*injection_parameters, dt=dt, T=T)
        except KeyboardInterrupt:
            raise
        except:
            injection_parameters[3] += 0.1
            print('increase by 0.1')
            continue
        break
    for m, pt in enumerate(tpls):
        shortened_wave = window_waveform(wave.copy(), T, dt, pt, window_size=2)
        waveforms[m,:shortened_wave.size] = shortened_wave
    snrs_out[k*grid_dims[-1]:(k+1)*grid_dims[-1]] = xp.sqrt(vectorised_snr2(waveforms.real,dt,psd_values,dfs) + vectorised_snr2(waveforms.imag,dt,psd_values,dfs))
    if not k % 1000:
        df['SNR'] = snrs_out.get()
        df.to_csv(outdir.format('4yr_grid6.csv'), index=False)    

df['SNR'] = snrs_out.get()
df.to_csv(outdir.format('4yr_grid6.csv'), index=False)


