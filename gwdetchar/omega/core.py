# coding=utf-8
# Copyright (C) Alex Urban (2019)
#
# This file is part of the GW DetChar python package.
#
# GW DetChar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GW DetChar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GW DetChar.  If not, see <http://www.gnu.org/licenses/>.

"""Core utilities for implementing omega scans
"""

from scipy.signal import butter

from gwpy.segments import Segment
from gwpy.signal.qtransform import q_scan

__author__ = 'Alex Urban <alexander.urban@ligo.org>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- basic utilities ----------------------------------------------------------

def highpass(series, f_low, order=12, analog=False, ftype='sos'):
    """High-pass a `TimeSeries` with a Butterworth filter

    Parameters
    ----------
    series : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to high-pass filter

    f_low : `float`
        lower cutoff frequency (Hz) of the filter

    order : `int`, optional
        number of taps in the filter, default: 12

    analog : `bool`, optional
        when True, return an analog filter, otherwise a digital filter is
        returned, default: False

    ftype : `str`, optional
        type of filter: numerator/denominator (`'ba'`), pole-zero (`'zpk'`), or
        second-order sections (`'sos'`), default: `'sos'`

    Returns
    -------
    hpseries : `~gwpy.timeseries.TimeSeries`
        the high-passed `TimeSeries`

    Notes
    -----
    This utility designs a Butterworth filter of order `order` with corner
    frequency `f_low / 1.5`, then applies this filter to the input.

    See Also
    --------
    scipy.signal.butter
    gwpy.timeseries.TimeSeries.filter
    """
    corner = f_low / 1.5
    fs = series.sample_rate.to('Hz').value
    hpfilt = butter(order, corner, btype='highpass', analog=analog,
                    output=ftype, fs=fs)
    hpseries = series.filter(hpfilt, filtfilt=True)
    return hpseries


def whiten(series, fftlength, overlap=None, method='median', window='hann',
           detrend='linear'):
    """Whiten a `TimeSeries` against its own ASD

    Parameters
    ----------
    series : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to whiten

    fftlength : `float`
        FFT integration length (in seconds) for ASD estimation

    overlap : `float`, optional
        seconds of overlap between FFTs, defaults to half the FFT length

    method : `str`, optional
        FFT-averaging method, default: ``'median'``,

    window : `str`, `numpy.ndarray`, optional
        window function to apply to timeseries prior to FFT,
        default: ``'hann'``
        see :func:`scipy.signal.get_window` for details on acceptable
        formats

    detrend : `str`, optional
        type of detrending to do before FFT, default: ``'linear'``

    Returns
    -------
    wseries : `~gwpy.timeseries.TimeSeries`
        a whitened version of the input data with zero mean and unit variance

    See Also
    --------
    gwpy.timeseries.TimeSeries.whiten
    """
    # get overlap window and whiten
    if overlap is None:
        overlap = fftlength / 2
    return series.whiten(fftlength=fftlength, overlap=overlap, window=window,
                         detrend=detrend, method=method).detrend(detrend)


def apply_coupling_functions(block, data, cf_file):
    import h5py
    from copy import copy
    from .config import OmegaChannel
    h5f = h5py.File(cf_file, 'r')
    new_channels = []
    for c in block.channels:
        channel_key = c.name.replace(':', '_').replace('-', '_')
        try:
            cf_data = h5f[channel_key + '/table'][()]
        except KeyError:
            print("no coupling function found for {}".format(c.name))
            new_channels.append(c)
            continue
        if '_MAG_' in c.name:
            cf_data = cf_data[cf_data['factor'] > 0]
        c_new = OmegaChannel(c.name + '_coupling', c.section, **c.params)
        series = apply_cf(data[c.name], cf_data)
        series.channel = c_new
        series.name = c_new.name
        data[c_new.name] = series
        new_channels.extend([c, c_new])
    h5f.close()
    block.channels = new_channels
    return block, data


def apply_cf(series, cf_data):
    """Apply coupling functions to time series
    """
    fft = series.fft()
    fft_freqs = fft.frequencies.value
    cf_interp = interp1d(fft_freqs, cf_data['frequency'], cf_data['factor_counts'])
    flag_interp = interp1d(fft_freqs, cf_data['frequency'], cf_data['flag'], kind='nearest')
    fft_pred = fft * cf_interp / 4000
    out = 2 * fft_pred.ifft()
    return out


def interp1d(x_out, x_in, y_in, kind='linear'):
    """Interpolation method that can also do nearest-value interpolation
    """
    import numpy as np
    if kind == 'linear':
        y_out = np.interp(x_out, x_in, y_in)
    elif kind == 'nearest':
        y_out = np.zeros(x_out.size, dtype=np.object)
        for i, x_value in enumerate(x_out):
            nearest = np.argmin(np.abs(x_in - x_value))
            y_out[i] = y_in[nearest]
    else:
        raise ValueError('invalid interpolation method ' + str(kind))
    return y_out


def get_waveform(graceid, ifo, approximant='SEOBNRv4_ROM', f_lower=10):
    """Get time-domain waveform of an event based on GraceDb parameters

    Parameters
    ----------
    graceid : str
    ifo : str, {'H1', 'L1'}
    approximant : str, optional
    f_lower : int, optional
    sample_rate : int, optional

    Returns
    -------
    waveform : `~gwpy.timeseries.TimeSeries` object
        Frequency time series f(t).
    """
    from ligo.gracedb.rest import GraceDb, HTTPError
    from pycbc.pnutils import get_inspiral_tf
    client = GraceDb()
    try:
        super_event = client.superevent(graceid).json()
    except HTTPError:
        eventid = graceid
    else:
        eventid = super_event['preferred_event']
    try:
        event = client.event(eventid).json()
    except HTTPError:
        raise ValueError("event {} not found".format(eventid))
    ifos = event['instruments'].split(',')
    if ifo in ifos:
        sngl_insp = event['extra_attributes']['SingleInspiral'][ifos.index(ifo)]
        end_time = sngl_insp['end_time'] + float(sngl_insp['end_time_ns']) * 1e-9
        start_time = end_time - sngl_insp['template_duration']
        times, freqs = get_inspiral_tf(
            end_time,
            sngl_insp['mass1'],
            sngl_insp['mass2'],
            sngl_insp['spin1z'],
            sngl_insp['spin2z'],
            f_lower,
            approximant=approximant
        )
        return times, freqs
    else:
        return None


# -- omega scans --------------------------------------------------------------

def conditioner(xoft, fftlength, overlap=None, resample=None, f_low=None,
                **kwargs):
    """Condition some input data for an omega scan

    Parameters
    ----------
    xoft : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to whiten

    fftlength : `float`
        FFT integration length (in seconds) for ASD estimation

    overlap : `float`, optional
        seconds of overlap between FFTs, defaults to half the FFT length

    resample : `int`, optional
        desired sampling rate (Hz) of the output if different from the input,
        default: no resampling

    f_low : `float`, optional
        lower cutoff frequency (Hz) of the filter, default: `None`

    **kwargs : `dict`, optional
        additional arguments to `omega.highpass`

    Returns
    -------
    wxoft : `~gwpy.timeseries.TimeSeries`
        a whitened version of the input data with zero mean and unit variance

    hpxoft : `~gwpy.timeseries.TimeSeries`
        high-passed version of the input data (returned only if `f_low` is
        not `None`)

    xoft : ``~gwpy.timeseries.TimeSeries`
        original (possibly resampled) version of the input data
    """
    if resample:
        xoft = xoft.resample(resample)
    # get whitened and high-passed data streams
    if f_low is None:
        wxoft = whiten(xoft, fftlength, overlap=overlap)
        return (wxoft, xoft)
    else:
        hpxoft = highpass(xoft, f_low, **kwargs)
        wxoft = whiten(hpxoft, fftlength, overlap=overlap)
        return (wxoft, hpxoft, xoft)


def primary(gps, length, hoft, fftlength, resample=None, f_low=None,
            **kwargs):
    """Condition the primary channel for use as a matched-filter

    Parameters
    ----------
    gps : `float`
        GPS time (seconds) of suspected transient

    length : `float`
        length (seconds) of the desired matched-filter

    hoft : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to whiten

    fftlength : `float`
        FFT integration length (in seconds) for ASD estimation

    resample : `int`, optional
        desired sampling rate (Hz) of the output if different from the input,
        default: no resampling

    f_low : `float`, optional
        lower cutoff frequency (Hz) of the filter, default: `None`

    **kwargs : `dict`
        additional keyword arguments to `omega.conditioner`

    Returns
    -------
    out : `~gwpy.timeseries.TimeSeries`
        the conditioned data stream
    """
    if f_low is None:
        out, _ = conditioner(hoft, fftlength, resample=resample)
    else:
        out, _, _ = conditioner(
            hoft, fftlength, resample=resample, f_low=f_low, **kwargs)
    return out.crop(gps - length/2, gps + length/2).taper()


def cross_correlate(xoft, hoft):
    """Cross-correlate two `TimeSeries` by matched-filter

    Parameters
    ----------
    xoft : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to analyze

    hoft : `~gwpy.timeseries.TimeSeries`
        a `TimeSeries` data to use as a matched-filter

    Returns
    -------
    out : `~gwpy.timeseries.TimeSeries`
        the output of a single phase matched-filter
    """
    # make sure series have consistent sample rates
    if hoft.sample_rate.value < xoft.sample_rate.value:
        xoft = xoft.resample(hoft.sample_rate.value)
    elif hoft.sample_rate.value > xoft.sample_rate.value:
        hoft = hoft.resample(xoft.sample_rate.value)
    out = xoft.correlate(hoft, window='hann')
    return out


def scan(gps, channel, xoft, fftlength, resample=None, fthresh=1e-10,
         search=0.5, nt=1400, nf=700, **kwargs):
    """Scan a channel for evidence of transients

    Parameters
    ----------
    gps : `float`
        the GPS time (seconds) to scan

    channel : `OmegaChannel`
        `OmegaChannel` object corresponding to this data stream

    xoft : `~gwpy.timeseries.TimeSeries`
        the `TimeSeries` data to analyze

    fftlength : `float`
        FFT integration length (in seconds) for ASD estimation

    resample : `int`, optional
        desired sampling rate (Hz) of the output if different from the input,
        default: no resampling

    fthresh : `float`, optional
        threshold on false alarm rate (Hz) for this channel to be considered
        interesting, default: 1e-10

    search : `float`, optional
        time window (seconds) around `gps` in which to find peak energies,
        default: 0.5

    nt : `int`, optional
        number of points on the time axis of the interpolated `Spectrogram`,
        default: 1400

    nf : `int`, optional
        number of points on the (log-sampled) frequency axis of the
        interpolated `Spectrogram`, default: 700

    **kwargs : `dict`, optional
        additional arguments to `omega.conditioner`

    Returns
    -------
    series : `tuple`
        an ordered collection of intermediate data products from this scan,
        including: the resampled `TimeSeries`, high-passed `TimeSeries`,
        whitened `TimeSeries`, whitened `QGram`, high-passed `QGram`,
        interpolated whitened `Spectrogram`, and interpolated high-passed
        `Spectrogram`
    """
    # condition data
    wxoft, hpxoft, xoft = conditioner(
        xoft, fftlength, resample=resample, f_low=channel.frange[0], **kwargs)
    # compute whitened Q-gram
    search = Segment(gps - search/2, gps + search/2)
    qgram, far = q_scan(
        wxoft, mismatch=channel.mismatch, qrange=channel.qrange,
        frange=channel.frange, search=search)
    if (far >= fthresh) and (not channel.always_plot):
        return None  # series is insignificant
    # compute raw Q-gram
    Q = qgram.plane.q
    rqgram, _ = q_scan(
        hpxoft, mismatch=channel.mismatch, qrange=(Q, Q),
        frange=qgram.plane.frange, search=search)
    # compute interpolated spectrograms
    tres = min(channel.pranges) / nt
    outseg = Segment(
        gps - max(channel.pranges)/2, gps + max(channel.pranges)/2)
    qspec = qgram.interpolate(tres=tres, fres=nf, logf=True,
                              outseg=outseg)
    rqspec = rqgram.interpolate(tres=tres, fres=nf, logf=True,
                                outseg=outseg)
    return (xoft, hpxoft, wxoft, qgram, rqgram, qspec, rqspec)
