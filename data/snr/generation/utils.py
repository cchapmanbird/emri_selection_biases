import numpy as np

try:
    import cupy as cp
    xp = cp
except (ModuleNotFoundError, ImportError):
    xp = np

import warnings

lisaD = 0.3  
lisaP = 2.0
MSUN_SI = 1.98848e30
YRSID_SI = 31558149.763545603
AU_SI = 149597870700.0
C_SI = 299792458.0
G_SI = 6.674080e-11
GMSUN = 1.3271244210789466e20
MTSUN_SI = 4.925491025873693e-06
MRSUN_SI = 1476.6250615036158
PC_SI = 3.0856775814913674e16
PI = 3.141592653589793238462643383279502884
PI_2 = 1.570796326794896619231321691639751442
PI_3 = 1.047197551196597746154214461093167628
PI_4 = 0.785398163397448309615660845819875721
SQRTPI = 1.772453850905516027298167483341145183
SQRTTWOPI = 2.506628274631000502415765284811045253
INVSQRTPI = 0.564189583547756286948079451560772585
INVSQRTTWOPI = 0.398942280401432677939946059934381868
GAMMA = 0.577215664901532860606512090082402431
SQRT2 = 1.414213562373095048801688724209698079
SQRT3 = 1.732050807568877293527446341505872367
SQRT6 = 2.449489742783178098197284074705891392
INVSQRT2 = 0.707106781186547524400844362104849039
INVSQRT3 = 0.577350269189625764509148780501957455
INVSQRT6 = 0.408248290463863016366214012450981898
F0 = 3.168753578687779e-08
Omega0 = 1.9909865927683788e-07
L_SI = 2.5e9
eorbit = 0.004824185218078991
ConstOmega = 1.99098659277e-7

lisaL = 2.5e9  # LISA's arm meters
lisaLT = lisaL / C_SI  # LISA's armn in sec

#### Noise levels
### Optical Metrology System noise
## Decomposition
Sloc = (1.7e-12) ** 2  # m^2/Hz
Ssci = (8.9e-12) ** 2  # m^2/Hz
Soth = (2.0e-12) ** 2  # m^2/Hz

## Global
Soms_d_all = {
    "Proposal": (10.0e-12) ** 2,
    "SciRDv1": (15.0e-12) ** 2,
    "MRDv1": (10.0e-12) ** 2,
}  # m^2/Hz

### Acceleration
Sa_a_all = {
    "Proposal": (3.0e-15) ** 2,
    "SciRDv1": (3.0e-15) ** 2,
    "MRDv1": (2.4e-15) ** 2,
}  # m^2/sec^4/Hz

def cornish_lisa_psd(f, sky_averaged=False, use_gpu=False, duration=1):
    """PSD from https://arxiv.org/pdf/1803.01944.pdf

    Power Spectral Density for the LISA detector assuming it has been active for a year.
    I found an analytic version in one of Niel Cornish's paper which he submitted to the arXiv in
    2018. I evaluate the PSD at the frequency bins found in the signal FFT.

    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf

    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    if sky_averaged:
        sky_averaging_constant = 20 / 3

    else:
        sky_averaging_constant = 1.0  # set to one for one source

    L = 2.5 * 10 ** 9  # Length of LISA arm
    f0 = 19.09 * 10 ** (-3)  # transfer frequency

    # Optical Metrology Sensor
    Poms = ((1.5e-11) * (1.5e-11)) * (1 + xp.power((2e-3) / f, 4))

    # Acceleration Noise
    Pacc = (
        (3e-15)
        * (3e-15)
        * (1 + (4e-4 / f) * (4e-4 / f))
        * (1 + xp.power(f / (8e-3), 4))
    )

    # constants for Galactic background after 1 year of observation
    if duration == 1:
        alpha = 0.171
        beta = 29/2
        k = 1020
        gamma = 1680
        f_k = 0.00215
    elif duration == 2:    
        alpha = 0.165
        beta = 299
        k = 1020
        gamma = 1340
        f_k = 0.00173
    elif duration == 4:
        alpha = 0.138
        beta = -221
        k = 521
        gamma = 1680
        f_k = 0.00113

    # Galactic background contribution
    Sc = (
        9e-45
        * xp.power(f, -7 / 3)
        * xp.exp(-xp.power(f, alpha) + beta * f * xp.sin(k * f))
        * (1 + xp.tanh(gamma * (f_k - f)))
    )

    # PSD
    PSD = (sky_averaging_constant) * (
        (10 / (3 * L * L))
        * (Poms + (4 * Pacc) / (xp.power(2 * np.pi * f, 4)))
        * (1 + 0.6 * (f / f0) * (f / f0))
        + Sc
    )

    return PSD

def window_waveform(wform, T, dt, tplunge, window_size=4):
    # window the waveform, such that the plunge is tplunge after the start of the observation. 
    # we implicitly assume the waveform plunges at wform[-1]
    start_ind = int((T - tplunge) * YRSID_SI / dt)
    window_samples = window_size * YRSID_SI / dt
    end_ind = min(wform.size, start_ind + window_samples)
    out_wform = wform[start_ind:end_ind]
    return out_wform

def vectorised_snr2(waveform_array, dt, psd_values, df_array):
    ffts = xp.fft.rfft(waveform_array, axis=1)[:,1:] * dt
    inners = 4 * xp.sum(xp.abs(ffts.conj() * ffts) * df_array / psd_values, axis=1)
    return inners

def inner_product(
        sig1,
        sig2,
        dt=None,
        df=None,
        f_arr=None,
        PSD_args=(),
        PSD_kwargs={},
        normalize=False,
        use_gpu=False,
):
    if use_gpu:
        xp = cp

    else:
        xp = np

    if df is None and dt is None and f_arr is None:
        raise ValueError("Must provide either df, dt or f_arr keyword arguments.")

    if isinstance(sig1, list) is False:
        sig1 = [sig1]

    if isinstance(sig2, list) is False:
        sig2 = [sig2]

    if len(sig1) != len(sig2):
        raise ValueError(
            "Signal 1 has {} channels. Signal 2 has {} channels. Must be equal.".format(
                len(sig1), len(sig2)
            )
        )

    if dt is not None:

        if len(sig1[0]) != len(sig2[0]):
            warnings.warn(
                "The two signals are two different lengths in the time domain. Zero padding smaller array."
            )

            length = len(sig1[0]) if len(sig1[0]) > len(sig2[0]) else len(sig2[0])

            sig1 = [xp.pad(sig, (0, length - len(sig1[0]))) for sig in sig1]
            sig2 = [xp.pad(sig, (0, length - len(sig2[0]))) for sig in sig2]

        length = len(sig1[0])

        freqs = xp.fft.rfftfreq(length, dt)[1:]

        ft_sig1 = [
            xp.fft.rfft(sig)[1:] * dt for sig in sig1
        ]  # remove DC / dt factor helps to cast to proper dimensionality
        ft_sig2 = [xp.fft.rfft(sig)[1:] * dt for sig in sig2]  # remove DC

    else:
        ft_sig1 = sig1
        ft_sig2 = sig2

        if df is not None:
            freqs = (xp.arange(len(sig1[0])) + 1) * df  # ignores DC component using + 1

        else:
            freqs = f_arr

    PSD_arr = xp.asarray(cornish_lisa_psd(freqs, *PSD_args, **PSD_kwargs))
    
    out = 0.0
    # assumes right summation rule
    x_vals = xp.zeros(len(PSD_arr))

    x_vals[1:] = xp.diff(freqs)
    x_vals[0] = x_vals[1]

    # account for hp and hx if included in time domain signal
    for temp1, temp2 in zip(ft_sig1, ft_sig2):
        y = xp.real(temp1.conj() * temp2) / PSD_arr  # assumes right summation rule

        out += 4 * xp.sum(x_vals * y)

    normalization_value = 1.0
    if normalize is True:
        norm1 = inner_product(
            sig1,
            sig1,
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
            use_gpu=use_gpu,
            normalize=False,
        )
        norm2 = inner_product(
            sig2,
            sig2,
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
            use_gpu=use_gpu,
            normalize=False,
        )

        normalization_value = np.sqrt(norm1 * norm2)

    elif isinstance(normalize, str):
        if normalize == "sig1":

            sig_to_normalize = sig1

        elif normalize == "sig2":
            sig_to_normalize = sig2

        else:
            raise ValueError(
                "If normalizing with respect to sig1 or sig2, normalize kwarg must either be 'sig1' or 'sig2'."
            )

        normalization_value = inner_product(
            sig_to_normalize,
            sig_to_normalize,
            dt=dt,
            df=df,
            f_arr=f_arr,
            PSD_args=PSD_args,
            PSD_kwargs=PSD_kwargs,
            normalize=False,
        )

    elif normalize is not False:
        raise ValueError("Normalize must be True, False, 'sig1', or 'sig2'.")

    out /= normalization_value
    return out


def snr(sig1, *args, data=None, use_gpu=False, **kwargs):
    if use_gpu:
        xp = cp

    else:
        xp = np

    if data is None:
        sig2 = sig1

    else:
        sig2 = data

    return xp.sqrt(inner_product(sig1, sig2, *args, use_gpu=use_gpu, **kwargs))
