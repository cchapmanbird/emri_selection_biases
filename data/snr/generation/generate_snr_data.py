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

outdir = '/home/christian/emri_selection_paper/snr_datasets/T2/{}'
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

total = int(1e5)
plunge_times_to_run = 5
overall_total = total * plunge_times_to_run

cols = ['logM', 'logq', 'a', 'e0', 'Y0', 'thetaS', 'phiS-phiK', 'thetaK']

lMs = np.random.uniform(np.log10(5e4),8,total)
mus = np.random.uniform(1,200,total)
lqs = np.log10(mus / 10**lMs)
avs = np.random.uniform(0.001, 0.999, total)
es = np.random.uniform(0.1, 0.5, total)
ys = np.random.uniform(0.5, 0.99, total) * (2*np.random.randint(2, size=int(total))-1)
qSs = np.arccos(2*np.random.uniform(0,1,size=total)-1)
tSs= np.random.uniform(0,2*np.pi,size=total)
qKs = np.arccos(2*np.random.uniform(0,1,size=total)-1)

event_params = np.vstack((lMs, lqs, avs, es, ys, qSs, tSs, qKs)).T
all_event_params = np.repeat(event_params,plunge_times_to_run,axis=0)
df = pd.DataFrame(all_event_params, columns=cols)

snrs_out = xp.zeros(overall_total)
tplunges = np.zeros(overall_total)

i = 0
waveforms = xp.zeros((plunge_times_to_run,nwavesamps), dtype=xp.complex64)
for k, paramset in enumerate(event_params):
    waveforms = xp.zeros((plunge_times_to_run,nwavesamps), dtype=xp.complex64)
    injection_parameters[0] = 10**paramset[0]
    injection_parameters[1] = 10**(paramset[0] + paramset[1])
    injection_parameters[2] = paramset[2]
    injection_parameters[4:6] = paramset[3:5]
    injection_parameters[7:10] = paramset[5:8]

    tplunge_vals = np.random.uniform(0,2,plunge_times_to_run)

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

            
    print(f'Event {k}, {injection_parameters[:6]}')
    wave = aak(*injection_parameters, dt=dt, T=T)
    for m, pt in enumerate(tplunge_vals):
        shortened_wave = window_waveform(wave.copy(), T, dt, pt, window_size=2)
        waveforms[m,:shortened_wave.size] = shortened_wave
    snrs_out[k*plunge_times_to_run:(k+1)*plunge_times_to_run] = xp.sqrt(vectorised_snr2(waveforms.real,dt,psd_values,dfs) + vectorised_snr2(waveforms.imag,dt,psd_values,dfs))
    tplunges[k*plunge_times_to_run:(k+1)*plunge_times_to_run] = tplunge_vals
    if not k % 1000:
        df['tplunge'] = tplunges
        df['SNR'] = snrs_out.get()
        df.to_csv(outdir.format('2yr_3.csv'), index=False)    

df['tplunge'] = tplunges
df['SNR'] = snrs_out.get()
df.to_csv(outdir.format('2yr_3.csv'), index=False)


