# Copyright 2018, Tom Eulenfeld, MIT license
"""
Calculate 2D, 1D, 0D bicoherence, bispectrum, polycoherence and polyspectrum
"""

from math import pi
import numpy as np
from numpy.fft import rfftfreq, rfft
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


def __get_norm(norm):
    if norm == 0 or norm is None:
        return None, None
    else:
        try:
            norm1, norm2 = norm
        except TypeError:
            norm1 = norm2 = norm
        return norm1, norm2


def __freq_ind(freq, f0):
    try:
        return [np.argmin(np.abs(freq - f)) for f in f0]
    except TypeError:
        return np.argmin(np.abs(freq - f0))


def __product_other_freqs(spec, indices, synthetic=(), t=None):
    p1 = np.prod([amplitude * np.exp(2j * np.pi * freq * t + phase)
                  for (freq, amplitude, phase) in synthetic], axis=0)
    p2 = np.prod(spec[:, indices[len(synthetic):]], axis=1)
    return p1 * p2


def _polycoherence_0d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    """Polycoherence between freqs and sum of freqs"""
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    ind = __freq_ind(freq, freqs)
    sum_ind = __freq_ind(freq, np.sum(freqs))
    spec = np.transpose(spec, [1, 0])
    p1 = __product_other_freqs(spec, ind, synthetic, t)
    p2 = np.conjugate(spec[:, sum_ind])
    coh = np.mean(p1 * p2, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 = np.mean(np.abs(p1) ** norm1 * np.abs(p2) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return coh


def _polycoherence_1d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    """
    Polycoherence between f1 given freqs and their sum as a function of f1
    """
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind2 = __freq_ind(freq, freqs)
    ind1 = np.arange(len(freq) - sum(ind2))
    sumind = ind1 + sum(ind2)
    otemp = __product_other_freqs(spec, ind2, synthetic, t)[:, None]
    temp = spec[:, ind1] * otemp
    temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh


def _polycoherence_1d_sum(data, fs, f0, *ofreqs, norm=2,
                          synthetic=(), **kwargs):
    """Polycoherence with fixed frequency sum f0 as a function of f1"""
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None]
    sumind = __freq_ind(freq, f0)
    ind1 = np.arange(np.searchsorted(freq, f0 - np.sum(ofreqs)))
    ind2 = sumind - ind1 - sum(ind3)
    temp = spec[:, ind1] * spec[:, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind, None])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh


def _polycoherence_2d(data, fs, *ofreqs, norm=2, flim1=None, flim2=None,
                      synthetic=(), **kwargs):
    """
    Polycoherence between freqs and their sum as a function of f1 and f2
    """
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.require(spec, 'complex64')
    spec = np.transpose(spec, [1, 0])  # transpose (f, t) -> (t, f)
    if flim1 is None:
        flim1 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    if flim2 is None:
        flim2 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    ind1 = np.arange(*np.searchsorted(freq, flim1))
    ind2 = np.arange(*np.searchsorted(freq, flim2))
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None, None]
    sumind = ind1[:, None] + ind2[None, :] + sum(ind3)
    temp = spec[:, ind1, None] * spec[:, None, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** norm1, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    del temp
    if norm is not None:
        coh = np.abs(coh, out=coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], freq[ind2], coh


def polycoherence(data, *args, dim=2, **kwargs):
    """
    Polycoherence between frequencies and their sum frequency

    Polycoherence as a function of two frequencies.

    |<prod(spec(fi)) * conj(spec(sum(fi)))>| ** n0 /
        <|prod(spec(fi))|> ** n1 * <|spec(sum(fi))|> ** n2

    i ... 1 - N: N=2 bicoherence, N>2 polycoherence
    < > ... averaging
    | | ... absolute value

    data: 1d data
    fs: sampling rate
    ofreqs: further positional arguments are fixed frequencies

    dim:
        2 - 2D polycoherence as a function of f1 and f2, ofreqs are additional
            fixed frequencies (default)
        1 - 1D polycoherence as a function of f1, at least one fixed frequency
            (ofreq) is expected
        'sum' - 1D polycoherence with fixed frequency sum. The first argument
            after fs is the frequency sum. Other fixed frequencies possible.
        0 - polycoherence for fixed frequencies
    norm:
        2 - return polycoherence, n0 = n1 = n2 = 2 (default)
        0 - return polyspectrum, <prod(spec(fi)) * conj(spec(sum(fi)))>
        tuple (n1, n2): general case with n0=2
    synthetic:
        used for synthetic signal for some frequencies,
        list of 3-item tuples (freq, amplitude, phase), freq must coincide
        with the first fixed frequencies (ofreq, except for dim='sum')
    flim1, flim2: for 2D case, frequency limits can be set
    **kwargs: are passed to scipy.signal.spectrogram. Important are the
        parameters nperseg, noverlap, nfft.
    """
    N = len(data)
    kwargs.setdefault('nperseg', N // 20)
    kwargs.setdefault('nfft', next_fast_len(N // 10))
    if dim == 0:
        f = _polycoherence_0d
    elif dim == 1:
        f = _polycoherence_1d
    elif dim == 'sum':
        f = _polycoherence_1d_sum
    elif dim == 2:
        f = _polycoherence_2d
    else:
        raise
    return f(data, *args, **kwargs)


def plot_polycoherence(freq1, freq2, bicoh):
    """
    Plot polycoherence (i.e. return values of polycoherence with dim=2)
    """
    df1 = freq1[1] - freq1[0]
    df2 = freq2[1] - freq2[0]
    freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
    freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2
    plt.figure()
    plt.pcolormesh(freq2, freq1, np.abs(bicoh))
    plt.xlabel('freq (Hz)')
    plt.ylabel('freq (Hz)')
    plt.colorbar()


def _plot_polycoherence_1d(freq, coh):
    plt.figure()
    plt.plot(freq, coh)
    plt.xlabel('freq (Hz)')


def _plot_signal(t, signal):
    plt.figure()
    plt.subplot(211)
    plt.plot(t, signal)
    plt.xlabel('time (s)')
    plt.subplot(212)
    ndata = len(signal)
    nfft = next_fast_len(ndata)
    freq = rfftfreq(nfft, t[1] - t[0])
    spec = rfft(signal, nfft) * 2 / ndata
    plt.plot(freq, np.abs(spec))
    plt.xlabel('freq (Hz)')
    plt.tight_layout()


def _test():
    N = 10001
    kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))
    t = np.linspace(0, 100, N)
    fs = 1 / (t[1] - t[0])
    s1 = np.cos(2 * pi * 5 * t + 0.2)
    s2 = 3 * np.cos(2 * pi * 7 * t + 0.5)
    s3 = 4 * np.cos(2 * pi * 1 * t + 0.1)
    s4 = 4 * np.cos(2 * pi * 9.5 * t + 0.7)
    s5 = 2 * np.cos(2 * pi * 0.02 * t + 1)
    np.random.seed(0)
    noise = 5 * np.random.normal(0, 1, N)

    # bicoherence
    signal = s1 + s2 + noise + 0.5 * s1 * s2
    _plot_signal(t, signal)
    plt.suptitle('signal and spectrum for bicoherence tests')
    print('bicoherence for f1=5Hz, f2=7Hz:',
          polycoherence(signal, fs, 5, 7, dim=0, **kw))
    print('bicoherence for f1=5Hz, f2=6Hz:',
          polycoherence(signal, fs, 5, 6, dim=0, **kw))

    result = polycoherence(signal, fs, **kw)
    plot_polycoherence(*result)
    plt.suptitle('bicoherence')

    result = polycoherence(signal, fs, norm=None, **kw)
    plot_polycoherence(*result)
    plt.suptitle('bispectrum')

    result = polycoherence(signal, fs, 5, dim=1, **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('bicoherence for f2=5Hz (column, expected 2Hz, 7Hz)')

    result = polycoherence(signal, fs, 12, dim='sum', **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('bicoherence for f1+f2=12Hz (diagonal, expected 5Hz, 7Hz)')

   # tricoherence
    signal = s2 + s3 + s4 + 0.1 * s2 * s3 * s4 + noise
    _plot_signal(t, signal)
    plt.suptitle('signal and spectrum for tricoherence tests')
    print('tricoherence for f1=1Hz, f2=7Hz, f3=9.5Hz:',
          polycoherence(signal, fs, 1, 7, 9.5, dim=0, **kw))

    result = polycoherence(signal, fs, 9.5, flim1=(0., 2), flim2=(6., 8), **kw)
    plot_polycoherence(*result)
    plt.suptitle('tricoherence with f3=9.5Hz')

    result = polycoherence(signal, fs, 9.5, norm=None, **kw)
    plot_polycoherence(*result)
    plt.suptitle('trispectrum with f3=9.5Hz')

    result = polycoherence(signal, fs, 1, 9.5, dim=1, **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('tricoherence for f2=1Hz, f3=9.5Hz')

    result = polycoherence(signal, fs, 17.5, 9.5, dim='sum', **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('tricoherence for f1+f2+f3=17.5Hz f3=9.5Hz')

   # tricoherence with synthetic signal
    signal = s2 + s3 + s5 - 0.5 * s2 * s3 * s5 + noise
    _plot_signal(t, signal)
    plt.suptitle('signal and spectrum for tricoherence tests with synthetic')
    synthetic = ((0.02, 10, 1), )
    print('tricoherence for f1=0.02Hz (synthetic)), f2=1Hz, f3=7Hz:',
          polycoherence(signal, fs, 0.02, 1, 7, dim=0,
                        synthetic=synthetic, **kw))
    result = polycoherence(signal, fs, 0.02, synthetic=synthetic, **kw)
    plot_polycoherence(*result)
    plt.suptitle('tricoherence with f3=0.02Hz (synthetic)')

    result = polycoherence(signal, fs, 0.02, synthetic=synthetic,
                           norm=None, **kw)
    plot_polycoherence(*result)
    plt.suptitle('trispectrum with f3=0.02Hz (synthetic)')

    result = polycoherence(signal, fs, 0.02, 7, dim=1,
                           synthetic=synthetic, **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('tricoherence for f2=0.02Hz (synthetic), f3=7Hz')

    result = polycoherence(signal, fs, 8.02, 0.02, dim='sum',
                           synthetic=synthetic, **kw)
    _plot_polycoherence_1d(*result)
    plt.suptitle('tricoherence for f1+f2+f3=8.02Hz f3=0.02Hz (synthetic)')

    plt.show()


if __name__ == '__main__':
    _test()
