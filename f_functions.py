#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import lmfit
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons, Cursor, SpanSelector, AxesWidget
import itertools
import warnings
warnings.filterwarnings(action='ignore')

#-------------------------------------------------------------------#
#                      functions from KLASSEZ                       #
#          Repository: https://github.com/MetallerTM/klassez        #
#-------------------------------------------------------------------#

def acme(data, m=1, a=5e-5):
    """
    Automated phase Correction based on Minimization of Entropy.
    This algorithm allows for automatic phase correction by minimizing the entropy of the m-th derivative of the spectrum, as explained in detail by L. Chen et.al. in Journal of Magnetic Resonance 158 (2002) 164-168.
    
    Defined the entropy of h as:
        S = - sum_j h_j ln(h_j)
    and 
        h = | R_j^(m) | / sum_j | R_j^(m) |
    where 
        R = Re{ spectrum * e^(-i phi) }
    and R^(m) is the m-th derivative of R, the objective function to minimize is:
        S + P(R)
    where P(R) is a penalty function for negative values of the spectrum.

    The phase correction is applied using processing.ps. The values p0 and p1 are fitted using Nelder-Mead algorithm.
    ----------
    Parameters:
    - data: 1darray
        Spectrum to be phased, complex
    - m: int
        Order of the derivative to be computed
    - a: float
        Weighting factor for the penalty function
    ----------
    Returns:
    - p0f: float
        Fitted zero-order phase correction, in degrees
    - p1f: float
        Fitted first-order phase correction, in degrees
    """

    def entropy(data):
        """
        Compute entropy of data.
        --------
        Parameters:
        - data: ndarray
            Input data
        --------
        Returns:
        - S: float
            Entropy of data
        """
        data_in = np.copy(data)
        if not data_in.all():
            zero_ind = np.flatnonzero(data_in==0)
            for i in zero_ind:
                data_in[i] = 1e-15

        return - np.sum(data_in * np.log(data_in))

    def mth_derivative(data, m):
        """
        Computes the m-th derivative of data by applying np.gradient m times.
        -------
        Parameters:
        - data: 1darray
            Input data
        - m: int
            Order of the derivative to be computed
        -------
        Returns:
        - pdata: 1darray
            m-th derivative of data
        """
        pdata = np.copy(data)
        for k in range(m):
            pdata = np.gradient(pdata)
        return pdata

    def penalty_function(data, a=5e-5):
        """
        F(y) is a function that is 0 for positive y and 1 otherwise.
        The returned value is 
            a * sum_j F(y_j) y_j^2
        --------
        Parameters:
        - data: 1darray
            Input data
        - a: float
            Weighting factor
        --------
        Returns:
        - p_fun: float
            a * sum_j F(y_j) y_j^2
        """
        signs = - np.sign(data)     # 1 for negative entries, -1 for positive entries
        p_arr = np.array([0 if j<1 else 1 for j in signs])  # replace all !=1 values in signs with 0
        p_fun = a * np.sum(p_arr * data**2)     # Make the sum
        return p_fun

    def f2min(param, data, m, a):
        """ Cost function for the fit. Applies the algorithm. """
        par = param.valuesdict()
        p0 = par['p0']
        p1 = par['p1']

        # Phase data and take real part
        Rp, *_ = ps(data, p0=p0, p1=p1)
        R = Rp.real

        # Compute the derivative and the h function
        Rm = np.abs(mth_derivative(R, m))
        H = np.sum(Rm)  # Normalization factor
        h = Rm / H

        # Calculate the penalty factor
        P = penalty_function(R, a)

        # Compute the residual
        res = entropy(h) + P
        return res

    if not np.iscomplexobj(data):
        raise ValueError('Input data is not complex.')

    # Define the parameters of the fit
    param = lmfit.Parameters()
    param.add('p0', value=0, min=-360, max=360)
    param.add('p1', value=0, min=-720, max=720)

    # Minimize using simplex method because the residue is a scalar
    minner = lmfit.Minimizer(f2min, param, fcn_args=(np.copy(data), m, a))
    result = minner.minimize(method='nelder', tol=1e-15)
    popt = result.params.valuesdict()

    return popt['p0'], popt['p1']

def ps(data, ppmscale=None, p0=None, p1=None, pivot=None):
    """
    Applies phase correction on the last dimension of data.
    The pivot is set at the center of the spectrum by default.
    Missing parameters will be inserted interactively.
    -------
    Parameters:
    - data: ndarray
        Input data
    - ppmscale: 1darray or None
        PPM scale of the spectrum. Required for pivot and interactive phase correction
    - p0: float
        Zero-order phase correction angle /degrees
    - p1: float
        First-order phase correction angle /degrees
    - pivot: float or None.
        First-order phase correction pivot /ppm. If None, it is the center of the spectrum.
    --------
    Returns:
    - datap: ndarray
        Phased data
    - final_values: tuple
        Employed values of the phase correction. (p0, p1, pivot)
    """
    if p0 is None and p1 is None:
        interactive = True
    elif p0 is None and p1 is not None:
        p0 = 0
    elif p1 is None and p0 is not None:
        p1 = 0

    if not np.iscomplexobj(data):
        raise ValueError('Data is not complex! Impossible to phase')
    
    p0 = p0 * np.pi / 180
    p1 = p1 * np.pi / 180
    size = data.shape[-1]
    pvscale = np.arange(size) / size
    if pivot is None:
        pv = 0.5
    else:
        pv = (ppmfind(ppmscale, pivot)[0] / size) 
    apod = np.exp(1j * (p0 + p1 * (pvscale - pv))).astype(data.dtype)
    datap = data * apod
    final_values = p0*180/np.pi, p1*180/np.pi, pivot

    return datap, final_values

def calcres(fqscale):
    """
    Calculates the frequency resolution of an axis scale, i.e. how many Hz is a "tick".
    --------
    Parameters:
    - fqscale : 1darray
        Scale to be processed
    -------
    Returns:
    - res: float
        The resolution of the scale
    """
    return np.abs(fqscale[1]-fqscale[0])

def set_fontsizes(ax, fontsize=10):
    """
    Automatically adjusts the fontsizes of all the figure elements.
    In particular:
    > title = fontsize
    > axis labels = fontsize - 2
    > ticks labels = fontsize - 3
    > legend entries = fontsize - 4
    --------
    Parameters:
    - ax: matplotlib.Subplot Object
        Subplot of interest
    - fontsize: float
        Starting fontsize
    -------
    """

    # ---------------------------------------------------------------------
    def _modify_legend(ax, **kwargs):
        """
        Copied from StackOverflow: 
            https://stackoverflow.com/questions/23689728/how-to-modify-matplotlib-legend-after-it-has-been-created
        """

        l = ax.legend_
        defaults = dict(
            loc = l._loc,
            numpoints = l.numpoints,
            markerscale = l.markerscale,
            scatterpoints = l.scatterpoints,
            scatteryoffsets = l._scatteryoffsets,
            prop = l.prop,
            borderpad = l.borderpad,
            labelspacing = l.labelspacing,
            handlelength = l.handlelength,
            handleheight = l.handleheight,
            handletextpad = l.handletextpad,
            borderaxespad = l.borderaxespad,
            columnspacing = l.columnspacing,
            ncol = l._ncols,
            mode = l._mode,
            fancybox = type(l.legendPatch.get_boxstyle())==matplotlib.patches.BoxStyle.Round,
            shadow = l.shadow,
            title = l.get_title().get_text() if l._legend_title_box.get_visible() else None,
            framealpha = l.get_frame().get_alpha(),
            bbox_to_anchor = l.get_bbox_to_anchor()._bbox,
            bbox_transform = l.get_bbox_to_anchor()._transform,
            #frameon = l._drawFrame,
            frameon = l.draw_frame,
            handler_map = l._custom_handler_map,
        )

        if "fontsize" in kwargs and "prop" not in kwargs:
            defaults["prop"].set_size(kwargs["fontsize"])

        ax.legend(**dict(list(defaults.items()) + list(kwargs.items())))
    # ---------------------------------------------------------------------

    # Set the dimensions
    title_font = fontsize
    label_font = fontsize - 2
    ticks_font = fontsize - 3
    legen_font = fontsize - 4

    ax.title.set_fontsize(title_font)                   # title
    ax.xaxis.label.set_fontsize(label_font)             # xlabel
    ax.yaxis.label.set_fontsize(label_font)             # xlabel
    # Ticks
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(ticks_font)
    # Offset text
    ax.xaxis.get_offset_text().set_size(ticks_font)
    ax.yaxis.get_offset_text().set_size(ticks_font)

    # Legend
    if ax.legend_ is not None:
        _modify_legend(ax, fontsize=legen_font)

def mathformat(ax, axis='y', limits=(-2,2)):
    """
    Apply exponential formatting to the given axis of the given figure panel. The offset text size is uniformed to the tick labels' size.
    -------
    Parameters:
    - ax: matplotlib.Subplot Object
        Panel of the figure to edit
    - axis: str
        'x', 'y' or 'both'.
    - limits: tuple
        tuple of ints that indicate the order of magnitude range outside which the exponential format is applied.
    """
    ax.ticklabel_format(axis=axis, style='scientific', scilimits=limits, useMathText=True)
    if axis=='y' or axis=='both':
        tmp = (ax.get_yticklabels())
        fontsize = tmp[0].get_fontsize()
        ax.yaxis.get_offset_text().set_size(fontsize)

    if axis=='x' or axis=='both':
        tmp = (ax.get_xticklabels())
        fontsize = tmp[0].get_fontsize()
        ax.xaxis.get_offset_text().set_size(fontsize)

def pretty_scale(ax, limits, axis='x', n_major_ticks=10):
    """
    This function computes a pretty scale for your plot. Calculates and sets a scale made of 'n_major_ticks' numbered ticks, spaced by 5*n_major_ticks unnumbered ticks. After that, the plot borders are trimmed according to the given limits.
    --------
    Parameters:
    - ax: matplotlib.AxesSubplot object
        Panel of the figure of which to calculate the scale
    - limits: tuple
        limits to apply of the given axis. (left, right)
    - axis: str
        'x' for x-axis, 'y' for y-axis
    - n_major_ticks: int
        Number of numbered ticks in the final scale. An oculated choice gives very pleasant results.
    """

    import matplotlib.ticker as TKR

    if axis=='x':
        ax.set_xlim(limits)
        sx, dx = ax.get_xlim()
    elif axis=='y':
        ax.set_ylim(limits)
        sx, dx = ax.get_ylim()
    else:
        raise ValueError('Unknown options for "axis".')

    # Compute major ticks
    steps = [1, 2, 4, 5, 10]
    majorlocs = TKR.MaxNLocator(nbins=n_major_ticks, steps=steps).tick_values(sx, dx)

    # Compute minor ticks manually because matplotlib is complicated
    ndivs = 5
    majorstep = majorlocs[1] - majorlocs[0]
    minorstep = majorstep / ndivs

    vmin, vmax = sx, dx
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    t0 = majorlocs[0]
    tmin = ((vmin - t0) // minorstep + 1) * minorstep
    tmax = ((vmax - t0) // minorstep + 1) * minorstep
    minorlocs = np.arange(tmin, tmax, minorstep) + t0

    # Set the computed ticks and update the limits
    if axis == 'x':
        ax.set_xticks(majorlocs)
        ax.set_xticks(minorlocs, minor=True)
        ax.set_xlim(sx,dx)
    elif axis == 'y':
        ax.set_yticks(majorlocs)
        ax.set_yticks(minorlocs, minor=True)
        ax.set_ylim(sx,dx)

def find_nearest(array, value):
    # Finds the value in 'array' which is the nearest to 'value'
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def ppmfind(ppm_1h, value):
    #from classf.py
    avgstep = np.abs((ppm_1h[0]-ppm_1h[1])/2)
    for i, delta in enumerate(ppm_1h):
        if delta >= value-avgstep and delta < value+avgstep:
            I = i
            V = ppm_1h[i]
            break
        else:
            continue

    return I, V

def trim_data(ppm_scale, y, sx, dx):
    # Trims the frequency scale and correspondant
    # 1D dataset 'y' from 'sx' (ppm) to 'dx' (ppm).
    # if "freq_scale" is already the ppm scale,
    # "ppm" must be set to "True" in order not to
    # trigger the conversion.
    SX = ppmfind(ppm_scale, sx)[0]
    DX = ppmfind(ppm_scale, dx)[0]
    ytrim = y[min(SX,DX):max(SX,DX)]
    xtrim = ppm_scale[min(SX,DX):max(SX,DX)]
    return xtrim, ytrim

def polyn(x, c):
    """
    Computes p(x), polynomion of degree n-1, where n is the number of provided coefficients.
    -------
    Parameters:
    - x :1darray
        Scale upon which to build the polynomion
    - c :list or 1darray
        Sequence of the polynomion coeffiecient, starting from the 0-th order coefficient
    -------
    Returns:
    - px :1darray
        Polynomion of degree n-1.
    """
    c = np.array(c)
    # Make the Vandermonde matrix of the x-scale
    T = np.array(
            [x**k for k in range(len(c))]
            ).T
    # Compute the polynomion via matrix multiplication
    px = T @ c
    return px

def em(data, lb, sw):
    """
    Applies an exponential multiplication to the data.
    --------
    Parameters:
    - data :ndarray
        Data to be processed
    - lb :float
        Line broadening factor
    - sw :float
        Spectral width of the spectrum
    --------
    Returns:
    - apod :ndarray
        Apodization function
    """
    if lb == 0:
        return np.ones_like(data, dtype=float)
    lb = lb / (2*sw)
    apod = np.exp(-np.pi * np.arange(data.shape[-1]) * lb).astype(data.dtype)
    return apod

def qsin(data, ssb):
    """
    Applies a squared-sine apodization to the data.
    --------
    Parameters:
    - data :ndarray
        Data to be processed
    - ssb :float
        squared-sine factor
    --------
    Returns:
    - apod :ndarray
        Apodization function
    """
    if ssb == 0:
        return np.ones_like(data, dtype=float)
    if ssb == 0 or ssb == 1:
        off = 0
    else:
        off = 1/ssb
    end = 1
    size = data.shape[-1]
    apod = np.power(np.sin(np.pi * off + np.pi * (end - off) * np.arange(size) / (size)).astype(data.dtype), 2).astype(data.dtype)
    return apod

def zf(data, size):
    """
    Zero-fills the data to the desired size.
    --------
    Parameters:
    - data :ndarray
        Data to be zero-filled
    - size :int
        Desired size of the zero-filled data
    --------
    Returns:
    - datazf :ndarray
        Zero-filled data
    """
    def zf_pad(data, pad):
        size = list(data.shape)
        size[-1] = int(pad)
        z = np.zeros(size, dtype=data.dtype)
        return np.concatenate((data, z), axis=-1)
    zpad = size - data.shape[-1]
    if zpad <= 0 :
        zpad = 0
    datazf = zf_pad(data, pad=zpad)
    return datazf

def calibration(ppm_scale, data, ppmsx, ppmdx, npoints=80, debug_fig=False):
    """
    Calibrates the spectra in 'data' with respect to the first one.
    The calibration is performed by shifting the spectra in 'data' with respect to the first one.
    The shift is calculated by minimizing the residue between the first spectrum and the others.
    --------
    Parameters:
    - ppm_scale : 1darray
        scale of the ppm axis
    - data : 2darray
        matrix with the spectra to be calibrated
    - ppmsx : float
        left limit of the calibration region
    - ppmdx : float
        right limit of the calibration region
    - npoints : int
        number of points to be used for the calibration
    - debug_fig : bool
        if True, shows a figure with the calibration process
    -------
    Returns:
    - shift_cal : 1darray
        array with the shifts in points
    - shift_cal_ppm : 1darray
        array with the shifts in ppm
    - data_roll : 2darray
        matrix with the calibrated spectra
    """

    print('Performing calibration...')

    def residue(param, ppm_scale, spettro0, spettro1, sx, dx, risultato=False):
        par = param.valuesdict()
        roll_spettro1 = np.roll(spettro1, int(par['shift']))
        if risultato==False:
            res = spettro0.real/max(spettro0.real)-roll_spettro1.real/max(roll_spettro1.real)
            return res[npoints:-npoints]
        else:
            if debug_fig:
                # FIGURA PER VEDERE COME VA LA CALIBRAZIONE
                fig = plt.figure()
                fig.set_size_inches(5.59,4.56)
                plt.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.90)
                ax = fig.add_subplot(1,1,1)
                ax.tick_params(labelsize=6.5)
                ax.plot(ppm_scale[sx:dx], spettro0.real/max(spettro0.real), lw=0.5, label='spectra_0')
                ax.plot(ppm_scale[sx:dx], roll_spettro1.real/max(roll_spettro1.real), lw=0.5, label='spectra_1')
                ax.plot(ppm_scale[sx:dx], spettro1.real/max(spettro1.real), lw=0.5, label='spectra_1 prima')
                ax.set_xlabel(r'$\delta \, ^1$H (ppm)', fontsize=8)
                ax.set_ylabel('Intensity (a.u.)', fontsize=8)
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
                ax.yaxis.get_offset_text().set_size(7)
                ax.invert_xaxis()
                ax.legend(fontsize=6)
                plt.show()
            return par['shift']

    sx,dx,_ = find_limits(ppmsx, ppmdx, ppm_scale)

    spettro0 = data[0,sx:dx]
    
    if npoints*2>len(spettro0):

        npoints = len(spettro0)//2
    
    shift_cal = []
    shift_cal_ppm = []
    for i in range(data.shape[0]):

        if i!=0:
            param = lmfit.Parameters()
            param.add('shift', value=0, max=npoints, min=-npoints)
            param['shift'].set(brute_step=1)
            spettro1 = data[i,sx:dx]
            minner = lmfit.Minimizer(residue, param, fcn_args=(ppm_scale, spettro0, spettro1, sx, dx))
            result = minner.minimize(method='brute', max_nfev=1000)#, xtol=1e-7, ftol=1e-7)
            popt = result.params
            # print('Fit report')
            # print(lmfit.fit_report(result))
            shift = residue(popt, ppm_scale, spettro0, spettro1, sx, dx, risultato=True)
            shift_cal.append(int(shift))
            shift_cal_ppm.append(shift*(ppm_scale[0]-ppm_scale[1]))
        else:
            shift_cal.append(0)
            shift_cal_ppm.append(0)

    print('...done')

    data_roll = np.zeros_like(data, dtype='float64')
    for i in range(data.shape[0]):
        data_roll[i,:] = np.roll(data[i,:], shift_cal[i])

    return shift_cal, shift_cal_ppm, data_roll

#=========================================================================================#

def find_limits(ppm1, ppm2, ppmscale):
    """
    Finds the limits of the interval between two given ppm values.
    --------
    Parameters:
    - ppm1 :float
        First ppm value
    - ppm2 :float
        Second ppm value
    - ppmscale :1darray
        PPM scale of the spectrum
    --------
    Returns:
    - sx :int
        Index corresponding to smaller ppm value
    - dx :int
        Index corresponding to greater ppm value
    - zero :float
        The ppm value of the two that is closer to zero
    """

    lim1 = ppmfind(ppmscale, ppm1)[0]
    lim2 = ppmfind(ppmscale, ppm2)[0]
    if lim1<lim2:
        sx = lim1
        dx = lim2
        zero = ppm2
    else:
        sx = lim2
        dx = lim1
        zero = ppm1

    return sx, dx, zero

def Interval(integral, x, y, center, I = 0.95):
    """
    Finds the interval of the spectrum that contains the given integral.
    --------
    Parameters:
    - integral :float
        Integral of the spectrum
    - x :1darray
        Scale of the spectrum
    - y :1darray
        Spectrum
    - center :int
        Index of the center of the interval
    - I :float
        Fraction of the integral to be found
    --------
    Returns:
    - lower_limit :int
        Index of the lower limit of the interval
    - upper_limit :int
        Index of the upper limit of the interval
    """
    #la scala è quella per punti
    target_integral = I * integral
    integral_so_far = 0
    lower_limit = center
    upper_limit = center
    for i in range(len(x)):
        if i==0:
            integral_so_far += y[center]
        else:
            integral_so_far += y[upper_limit] + y[lower_limit]
        if integral_so_far >= target_integral:
            break
        lower_limit -= 1
        upper_limit += 1
    return lower_limit, upper_limit

def error_calc_num(ppm_scale, res_tot, lor, Int, sx, dx, confidence=0.90):
    """
    Numerical error calculation for generic lineshape.
    --------
    Parameters:
    - ppm_scale :1darray
        PPM scale of the spectrum
    - res_tot :1darray
        Residuals of the fit
    - lor :1darray
        Lorentzian lineshape of the spectrum
    - Int :float
        Integral of the spectrum
    - shift :float
        Shift of the spectrum
    - sx :int
        Index of the lower limit of the interval
    - dx :int
        Index of the upper limit of the interval
    - confidence :float
        Confidence level of the error
    --------
    Returns:
    - Err_tot :float
        Total error of the spectrum
    """

    point_scale = np.arange(0,len(ppm_scale),1)
    if np.array_equal(lor, np.zeros_like(lor)):
        weights = np.ones_like(lor)
    else:
        weights = lor
    center = ppmfind(ppm_scale, np.average(ppm_scale, weights=weights))[0]
    lower, upper = Interval(Int, point_scale, lor, center, confidence)
    hole_dx = 0
    hole_sx = 0
    if sx>lower:
        hole_sx = np.trapz(lor[lower:sx])/Int
        lower = np.copy(sx)   
    if dx<upper:
        hole_dx = np.trapz(lor[dx:upper])/Int
        upper = np.copy(dx)
    Err_tot = np.trapz(np.abs(res_tot[lower-sx:upper-sx].real))
    return Err_tot + hole_dx*Err_tot + hole_sx*Err_tot

def input1_gen(CI):
    ##INPUT1 GENERATION##
    var = input("Write new input1? ([y]|n) ")
    if var == 'y' or var == '':
        filename1 = input("Write new input1 filename: ")
        CI.write_input_INT(filename1)
        print('New input1 saved in '+color_term.BOLD+filename1+color_term.END+' in current directory')
    elif var == 'n':
        filename1 = input("Type input1 filename: ")
    else:
        print('\nINVALID INPUT')
        exit()
    return filename1

def input2_gen(CI, matrix, acqupars, procpars, completebsl=False):
    ##INPUT2 GENERATION##
    var = input("Write new input2? ([y]|n) ")
    if var == 'y' or var == '':
        filename2 = input("Write new input2 filename: ")
        CI.interactive_iguess(matrix[:,:-1], filename2, acqupars, procpars, completebsl=completebsl)
        print('New input2 saved in '+color_term.BOLD+filename2+color_term.END+' in current directory')
    elif var == 'n':
        filename2 = input("Type input2 filename: ")
    else:
        print('\nINVALID INPUT')
        exit()
    return filename2

def t_lorentzian(t, v, s, A=1, phi=0):
    """
    Lorentzian lineshape function.
    --------
    Parameters:
    - t :1darray
        Time scale
    - v :float
        Frequency of the lineshape
    - s :float
        full width half maximum of the lineshape
    - A :float
        Amplitude of the lineshape (as ln(2))
    - phi :float
        Phase of the lineshape
    --------
    Returns:
    - S :1darray
        Lorentzian lineshape
    """
    S = A * np.exp(1j*phi) * np.exp((1j *2*np.pi *v * t)-(t*s/2)) #!!!!!!!!!!!!!!!!!!!
    return S

def t_gaussian_v(t, v, s, A=1, phi=0):
    """
    Gaussian lineshape function (for voigtian lineshape).
    --------
    Parameters:
    - t :1darray
        Time scale
    - v :float
        Frequency of the lineshape
    - s :float
        full width half maximum of the lineshape
    - A :float
        Amplitude of the lineshape (frection of gaussian)
    - phi :float
        Phase of the lineshape
    --------
    Returns:
    - S :1darray
        Gaussian lineshape
    """
    S = np.exp(1j*phi) * np.exp((1j*2*np.pi*v*t) - A*((t**2)*(s**2)/2))
    return S

def t_lorentzian_v(t, v, s, A=1, phi=0):
    """
    Lorentzian lineshape function (for voigtian lineshape).
    --------
    Parameters:
    - t :1darray
        Time scale
    - v :float
        Frequency of the lineshape
    - s :float
        full width half maximum of the lineshape
    - A :float
        Amplitude of the lineshape (fraction of lorentzian)
    - phi :float
        Phase of the lineshape
    --------
    Returns:
    - S :1darray
        Lorentzian lineshape
    """
    S = np.exp(1j*phi) * np.exp((1j*2*np.pi*v*t) - A*((t*s/2)))
    return S

def t_voigt(t, v, s, A=1, phi=0, x_g=0):
    """
    Voigt lineshape function.
    --------
    Parameters:
    - t :1darray
        Time scale
    - v :float
        Frequency of the lineshape
    - s :float
        full width half maximum of the lineshape
    - A :float
        Amplitude of the lineshape
    - phi :float
        Phase of the lineshape
    - x_g :float
        Fraction of gaussian lineshape
    --------
    Returns:
    - S :1darray
        Voigt lineshape
    """
    S = A * np.exp(1j*phi) * (t_gaussian_v(t, v/2, s/2.355, A=x_g) * t_lorentzian_v(t, v/2, s, A=1-x_g))   
    return S

def ft(fid, td = 65536, dw = 1e-6, o1p = 4.7, sfo1 = 1200):
    """
    Fourier transform of the FID.
    --------
    Parameters:
    - fid :1darray
        FID to be transformed
    - td :int
        Number of points in the time domain
    - dw :float
        Dwell time of the FID
    - o1p :float
        Carrier frequency of the FID
    - sfo1 :float
        Spectrometer frequency
    --------
    Returns:
    - data :1darray
        Transformed data
    """
    data = np.zeros_like(fid)
    fid[0] /= 2
    data = np.fft.fftshift(np.fft.fft(fid))
    freq = np.zeros_like(fid)
    freq = np.fft.fftshift(np.fft.fftfreq(td, d=dw)) / sfo1 + o1p
    return data, freq

def ift(data):
    """
    Inverse Fourier transform of the data.
    --------
    Parameters:
    - data :1darray
        Data to be transformed
    --------
    Returns:
    - fid :1darray
        Inverse transformed data
    """
    fid = np.zeros_like(data)
    fid = np.fft.ifft(np.fft.ifftshift(data))
    fid[0] *= 2
    return fid

def param_acq(ngdic):
    dict = {
        'dw' : 1/(ngdic['acqus']['SW_h']),
        'TD' : int(ngdic['acqus']['TD'])//2,
        'o1' : ngdic['acqus']['O1'],
        'SFO1' : ngdic['acqus']['SFO1'],
        'o1p' : ngdic['acqus']['O1']/ngdic['acqus']['SFO1'],
        'SWp' : ngdic['acqus']['SW'],
        'SW' : ngdic['acqus']['SW_h'],
        'GRPDLY' : ngdic['acqus']['GRPDLY'],
        'P1' : ngdic['acqus']['P'][1]*1e-6,
        'omega_nut': 1/(4*ngdic['acqus']['P'][1]*1e-6),
        'DE' : 0, #ngdic['acqus']['DE']*1e-6,
        'BF1' : ngdic['acqus']['BF1']
    }
    return dict 

def param_pr(ngdic):
    dict = {
        'SI' : int(ngdic['procs']['SI']),
        'SF' : ngdic['procs']['SF'],
        'SR' : (ngdic['procs']['SF']-ngdic['acqus']['BF1'])*1e6,
        'LB' : ngdic['procs']['LB'],
        'SSB' : ngdic['procs']['SSB'],
        'OFFSET' : ngdic['procs']['OFFSET']
    }
    return dict

class create_input():

    def __init__(self, ppm_scale, datap):
        self.pscale = ppm_scale
        self.datap = datap

    def select_regionsandpeaks(self):

        """
        Interactively select the slices that will be used in the fitting routine.
        ---------------
        Parameters:
        - ppm_scale: 1darray
            ppm scale of the spectrum
        - spectrum: 1darray
            Spectrum of the mixture
        --------------
        Returns:
        - regions: list of tuple
            Limits, in ppm
        """

        #colors for multiplets region:
        colormap = plt.cm.hsv
        num_colors = 10
        colors = [colormap(i / num_colors) for i in range(num_colors)]

        ppm_scale = self.pscale
        spectrum = self.datap

        ## SLOTS
        def onselect1(xmin, xmax):
            """ Print the borders of the span selector """
            span.set_visible(True)
            text = f'{xmax:-6.3f}:{xmin:-6.3f}'
            span_limits.set_text(text)
            plt.draw()
            pass

        def add(event):
            """ Add to the list """
            nonlocal colors

            if not multiplicity:
                # if len(span_limits.get_text()) == 0 or span.get_visible is False:
                #     return 
                lims = np.around(span.extents, 3)
                return_list.append((max(lims), min(lims)))
                ax.axvspan(*lims, facecolor='tab:green', edgecolor='g', alpha=0.2)
                text = f'{max(lims):-6.3f}:{min(lims):-6.3f}'
                output = output_text.get_text()
                output_text.set_text('\n'.join([output, text]))
                span_limits.set_text('')
                span.set_visible(False)
            else:
                if grouped_s:
                    print(grouped_s)
                    grouped.append(list(itertools.chain(*grouped_s)))
                    for xx in list(itertools.chain(*grouped_s)):
                        ax.axvline(xx, linestyle='--', c=colors[-1], lw=1)
                    del colors[-1]
                    grouped_s.clear()
                    print(grouped)
                    span_limits.set_text('')
                    span.set_visible(False)
                
                for q in span_list:
                    q.set_visible(False)  # Set the element to invisible
                    del q  # Delete the element
                span_list.clear()
            plt.draw()

        def save(event):
            plt.close()
            return return_list

        def increase_zoom(event):
            nonlocal lvlstep
            if lvlstep > 1:
                lvlstep = 1
            else:
                lvlstep += 0.04

        def decrease_zoom(event):
            nonlocal lvlstep
            if lvlstep < 1e-3:
                lvlstep = 1e-3
            else:
                lvlstep -= 0.02

        def on_click(event):
            if A_flag:
                on_click_A(event)
            else:
                on_click_notA(event)

        def radio_val(label):
            nonlocal A_flag
            if label=='true':
                A_flag = 1
            if label=='false':
                A_flag = 0

        def onselect2(xmin, xmax):

            grouped_s.append([v_A[i] for i in range(len(v_A)) if v_A[i]>xmin and v_A[i]<xmax])
            span_m = ax.axvspan(xmin, xmax, color='b', alpha=0.3)
            span.set_visible(True)
            plt.draw()
            
            span_list.append(span_m)

        def on_click_A(event):
            x = event.xdata
            if x is not None and x != 0:
                ix = find_nearest(ppm_scale, x)
                if event.dblclick and str(event.button) == '1': #'MouseButton.LEFT':
                    if ix not in v_A:
                        v_A.append(ix)
                        dotvline1.append(ax.axvline(ix, c='r', lw=0.4))
                if event.dblclick and str(event.button) == '0': #'MouseButton.RIGHT':
                    if ix in v_A:
                        i = v_A.index(ix)
                        v_A.remove(ix)
                        killv = dotvline1.pop(i)
                        killv.remove()
            fig.canvas.draw()

        def on_click_notA(event):
            x = event.xdata
            if x is not None and x != 0:
                ix = find_nearest(ppm_scale, x)
                if event.dblclick and str(event.button) == '1': #'MouseButton.LEFT':
                    if ix not in v_notA:
                        v_notA.append(ix)
                        dotvline2.append(ax.axvline(ix, c='k', lw=0.4))
                if event.dblclick and str(event.button) == '0': #'MouseButton.RIGHT':
                    if ix in v_notA:
                        i = v_notA.index(ix)
                        v_notA.remove(ix)
                        killv = dotvline2.pop(i)
                        killv.remove()
            fig.canvas.draw()

        def on_scroll(event):
            nonlocal scale_factor
            if flags:
                if event.button == 'up':
                    scale_factor += lvlstep
                if event.button == 'down':
                    scale_factor += -lvlstep
                if scale_factor < 0:
                    scale_factor = 0
            spect.set_ydata(S.real * scale_factor)
            fig.canvas.draw()

        Onselect = onselect1
        Colorspan = 'tab:red'

        def switch_onselect(event):
            nonlocal Onselect
            nonlocal Colorspan
            nonlocal multiplicity

            if event.key == 'm':
                multiplicity = not multiplicity
                Onselect = onselect2 if Onselect == onselect1 else onselect1
                Colorspan = 'tab:blue' if Colorspan == 'tab:red' else 'tab:red'
                mode_box.set_facecolor('tab:blue' if Onselect == onselect2 else 'tab:green')
                mode_text.set_text('M' if Onselect == onselect2 else 'I')
                span.__dict__['onselect'] = Onselect
                span.set_props(color=Colorspan)
                span.set_handle_props(color=Colorspan)
                plt.draw()

        #----------------------------------------------------------------------------------------------------

        flags = 1
        lvlstep = 0.1
        scale_factor = 1
        left = max(ppm_scale)
        right = min(ppm_scale)
        dotvline1 = []
        dotvline2 = []
        A_flag = 1
        v_A = []
        v_notA = []
        grouped_s = []
        grouped = []
        multiplicity = False
        span_list = []

        # Shallow copy of the spectrum
        S = np.copy(spectrum.real)
        # Placeholder for return values
        return_list = []

        # Make the figure
        fig = plt.figure()
        fig.set_size_inches((15, 9))
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.10, right=0.80, top=0.80, bottom=0.10)

        # Make boxes for widgets
        output_box = plt.axes([0.875, 0.100, 0.10, 0.70])
        output_box.set_facecolor('0.985')       # Grey background
        output_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        add_box = plt.axes([0.815, 0.820, 0.05, 0.06])
        save_box = plt.axes([0.815, 0.100, 0.05, 0.08])
        mode_box = plt.axes([0.975-0.032, 0.9, 0.032, 0.06])
        mode_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        mode_box.set_facecolor('tab:green')
        mode_text = mode_box.text(0.5, 0.5, 'I', ha='center', va='center', transform=mode_box.transAxes, fontsize=16)


        # Make widgets
        add_button = Button(add_box, 'ADD', hovercolor='0.975')
        save_button = Button(save_box, 'SAVE\nAND\nEXIT', hovercolor='0.975')
        radio = RadioButtons(plt.axes([0.815, 0.9, 0.1, 0.1]), ['true', 'false'], active=0)  #x,y,largo,alto

        # Plot the spectrum
        spect, = ax.plot(ppm_scale, S, c='k', lw=1.2, label='Mixture')

        ax.set_title('Drag the mouse to draw a region. Press "ADD" to add it to the list.\n'
                     'Double click with left button on the spectrum to add peaks. Double click with right button to remove peaks.\n'
                     'Press "M" to activate the multiplicity selection and drag the mouse over the multiplet components.\n'
                     'Then press "ADD" to save that multiplet.')
        
        # Add axes labels
        ax.set_xlabel(r'$\delta$ (ppm)')
        ax.set_ylabel('Intensity (a.u.)')

        ax.set_xlim(max(ppm_scale), min(ppm_scale))
        mathformat(ax)
        set_fontsizes(ax, 14)

        fig.canvas.mpl_connect('key_press_event', switch_onselect)
        # Declare span selector

        span = SpanSelector(
                ax,
                Onselect,
                'horizontal', 
                useblit=True,
                props=dict(alpha=0.2, facecolor=Colorspan),
                interactive=True,
                drag_from_anywhere=True,
                )
       
        cursor = Cursor(ax, useblit=True, color='red', linewidth=0.4, horizOn=False, vertOn=True)
        mouse = fig.canvas.mpl_connect('button_press_event', on_click)
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)


        # Connect widgets to the slots
        add_button.on_clicked(add)
        save_button.on_clicked(save)
        radio.on_clicked(radio_val)

        # Make output text
        span_limits = plt.text(0.925, 0.85, '', ha='center', va='center', transform=fig.transFigure, fontsize=11, color='tab:red') 
        output_text = output_box.text(0.5, 1.025, '\n', ha='center', va='top', color='tab:green', fontsize=11)

        plt.show()

        return return_list, v_A, v_notA, list(filter(None, grouped))

    def select_regions(self):

        """
        Interactively select the slices that will be used in the fitting routine.
        ---------------
        Parameters:
        - ppm_scale: 1darray
            ppm scale of the spectrum
        - spectrum: 1darray
            Spectrum of the mixture
        --------------
        Returns:
        - regions: list of tuple
            Limits, in ppm
        """

        ppm_scale = self.pscale
        spectrum = self.datap

        ## SLOTS
        def onselect(xmin, xmax):
            """ Print the borders of the span selector """
            span.set_visible(True)
            text = f'{xmax:-6.3f}:{xmin:-6.3f}'
            span_limits.set_text(text)
            plt.draw()
            pass

        def add(event):
            """ Add to the list """
            nonlocal return_list
            nonlocal err_list

            # # Do nothing if the span selector is invisible/not set
            # if len(span_limits.get_text()) == 0 or span.get_visible is False:
            #     return
            # Get the current limits rounded to the third decimal figure
            lims = np.around(span.extents, 3)
            # Append these values to the final list, in the correct order
            if not flag_err:
                return_list.append((max(lims), min(lims)))
                ax.axvspan(*lims, facecolor='tab:green', edgecolor='g', alpha=0.2)
            else:
                err_list = (max(lims), min(lims))
                ax.axvspan(*lims, facecolor='tab:blue', edgecolor='b', alpha=0.2)

            # Write the limits in the output box
            text = f'{max(lims):-6.3f}:{min(lims):-6.3f}'
            output = output_text.get_text()
            output_text.set_text('\n'.join([output, text]))
            # Reset the interactive text (the red one)
            span_limits.set_text('')
            # Turn the span selector invisible
            span.set_visible(False)
            plt.draw()

        def save(event):
            plt.close()
        
        def onselect_err(xmin, xmax):
            """ Print the borders of the span selector """
            span.set_visible(True)
            text = f'{xmax:-6.3f}:{xmin:-6.3f}'
            span_limits.set_text(text)
            plt.draw()
            pass

        def fork_span(xmin, xmax):
            nonlocal flag_err
            if flag_err:
                onselect_err(xmin, xmax)
            else:
                onselect(xmin, xmax)

        def switch_onselect(event):
            nonlocal flag_err
            if event.key == 'e':
                flag_err = not flag_err
                if flag_err:
                    mode_box.set_facecolor('tab:blue')
                    span.set_props(color='tab:blue')
                    span.set_handle_props(color='tab:blue')
                    mode_text.set_text('E')
                else:
                    mode_box.set_facecolor('tab:green')
                    span.set_props(color='tab:green')
                    span.set_handle_props(color='tab:green')
                    mode_text.set_text('I')
                plt.draw()

        def increase_zoom(event):
            nonlocal lvlstep
            if lvlstep > 1:
                lvlstep = 1
            else:
                lvlstep += 0.04

        def decrease_zoom(event):
            nonlocal lvlstep
            if lvlstep < 1e-3:
                lvlstep = 1e-3
            else:
                lvlstep -= 0.02

        def on_scroll(event):
            nonlocal scale_factor
            if flags:
                if event.button == 'up':
                    scale_factor += lvlstep
                if event.button == 'down':
                    scale_factor += -lvlstep
                if scale_factor < 0:
                    scale_factor = 0
            spect.set_ydata(S.real * scale_factor)
            fig.canvas.draw()



        #----------------------------------------------------------------------------------------------------
        flags = 1
        lvlstep = 0.1
        scale_factor = 1
        flag_err = False

        # Shallow copy of the spectrum
        S = np.copy(spectrum.real)
        # Placeholder for return values
        return_list = []
        err_list = None

        # Make the figure
        fig = plt.figure()
        fig.set_size_inches((15, 8))
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.10, right=0.80, top=0.90, bottom=0.10)
        
        # Make boxes for widgets
        output_box = plt.axes([0.875, 0.100, 0.10, 0.70])
        output_box.set_facecolor('0.985')       # Grey background
        output_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        add_box = plt.axes([0.815, 0.820, 0.05, 0.06])
        save_box = plt.axes([0.815, 0.100, 0.05, 0.08])
        mode_box = plt.axes([0.975-0.032, 0.820+0.06, 0.032, 0.06])
        mode_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        mode_box.set_facecolor('tab:green')
        mode_text = mode_box.text(0.5, 0.5, 'I', ha='center', va='center', transform=mode_box.transAxes, fontsize=16)

        # Make widgets
        add_button = Button(add_box, 'ADD', hovercolor='0.975')
        save_button = Button(save_box, 'SAVE\nAND\nEXIT', hovercolor='0.975')

        # Plot the spectrum
        spect, = ax.plot(ppm_scale, S, c='k', lw=1.2, label='Mixture')

        ax.set_title(r'Drag the mouse to draw a region. Press "ADD" to add it to the list.'+'\n'+r'Press "E" to select the region for error evaluation.')
        # Add axes labels
        ax.set_xlabel(r'$\delta$ (ppm)')
        ax.set_ylabel(r'Intensity (a.u.)')

        # Fancy adjustments
        # pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), 'x')
        # pretty_scale(ax, ax.get_ylim(), 'y')
        ax.set_xlim(max(ppm_scale), min(ppm_scale))
        mathformat(ax)
        set_fontsizes(ax, 14)

        # Declare span selector
        span = SpanSelector(
                ax,
                onselect,
                'horizontal', 
                useblit=True,
                props=dict(alpha=0.2, facecolor='tab:red'),
                interactive=True,
                drag_from_anywhere=True,
                )

        # Connect widgets to the slots
        add_button.on_clicked(add)
        save_button.on_clicked(save)
        fig.canvas.mpl_connect('key_press_event', switch_onselect)
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

        # Make output text
        span_limits = plt.text(0.925, 0.85, '', ha='center', va='center', transform=fig.transFigure, fontsize=11, color='tab:red') 
        output_text = output_box.text(0.5, 1.025, '\n', ha='center', va='top', color='tab:green', fontsize=11)

        plt.show()

        return_list = sorted(return_list, key=lambda x: x[0])

        return return_list, err_list

    def make_iguess(self, limits=None, true_limits=None):
        """
        Compute the initial guess for the quantitative fit of 1D NMR spectrum in an interactive manner.
        -------
        Parameters:
        - S : 1darray
            Spectrum to be fitted
        - ppm_scale : 1darray
            Self-explanatory
        - limits : tuple or None
            Trim limits for the spectrum (left, right). If None, the whole spectrum is used.
        -------
        Returns:
        - C_f : 1darray or False
            Coefficients of the polynomion to be used as baseline correction. If the 'baseline' checkbox in the interactive figure panel is not checked, C_f is False.
        """

        ppm_scale = self.pscale
        S = self.datap.real

        N = S.shape[-1]

        # Set limits according to rev
        if limits is None:
            limits = [max(ppm_scale), min(ppm_scale)]

        # Get index for the limits
        lim1 = ppmfind(ppm_scale, limits[0])[0]
        lim2 = ppmfind(ppm_scale, limits[1])[0]

        sl = slice(min(lim1, lim2), max(lim1, lim2))

        # make boxes for widgets
        poly_box = plt.axes([0.72, 0.10, 0.10, 0.3])
        su_box = plt.axes([0.815, 0.825, 0.08, 0.075])
        giu_box = plt.axes([0.894, 0.825, 0.08, 0.075])
        save_box = plt.axes([0.7, 0.825, 0.085, 0.04])
        reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])

        # Make widgets
        #   Buttons
        up_button = Button(su_box, '$\\uparrow$', hovercolor = '0.975')
        down_button = Button(giu_box, '$\\downarrow$', hovercolor = '0.975')
        save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
        reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')

        #   Radio
        poly_name = ['A', 'B', 'C', 'D', 'E']
        poly_radio = RadioButtons(poly_box, poly_name, activecolor='tab:orange')       # Polynomion

        # Create variable for the 'active' status
        stats = np.zeros(len(poly_name))
        #    a   b   c   d   e
        stats[0] = 1

        # Initial values
        #   Polynomion coefficients
        C = np.zeros(len(poly_name))

        #   Increase step for the polynomion (order of magnitude)
        om = np.zeros(len(poly_name))

        # Functions connected to the widgets
        def statmod(label):
            # Sets 'label' as active modifying 'stats'
            nonlocal stats
            if label in poly_name:    # if baseline
                stats = np.zeros(len(poly_name))
                for k, L in enumerate(poly_name):
                    if label == L:
                        stats[k] = 1
            update(0)       # Call update to redraw the figure

        def roll_up_p(event):
            # Increase polynomion with mouse scroll
            nonlocal C
            for k in range(len(poly_name)):
                if stats[k]:
                    C[k]+=10**om[k]

        def roll_down_p(event):
            # Decrease polynomion with mouse scroll
            nonlocal C
            for k in range(len(poly_name)):
                if stats[k]:
                    C[k]-=10**om[k]

        def up_om(event):
            # Increase the om of the active coefficient by 1
            nonlocal om
            for k in range(len(poly_name)):
                if stats[k]:
                    om[k] += 1

        def down_om(event):
            # Decrease the om of the active coefficient by 1
            nonlocal om
            for k in range(len(poly_name)):
                if stats[k]:
                    om[k] -= 1

        def switch_up(event):
            # Fork function for mouse scroll up
            up_om(event)

        def switch_down(event):
            # Fork function for mouse scroll down
            down_om(event)

        def on_scroll(event):
            # Mouse scroll
            if event.button == 'up':
                roll_up_p(event)
            elif event.button == 'down':
                roll_down_p(event)
            update(0)


        # polynomion
        x = np.linspace(0, 1, ppm_scale[sl].shape[-1])[::-1]
        y = np.zeros_like(x)

        # Initial figure
        fig = plt.figure(1)
        fig.set_size_inches(15,8)
        plt.subplots_adjust(bottom=0.15, top=0.90, left=0.1, right=0.65)
        ax = fig.add_subplot(1,1,1)
        ax.set_title("Scroll the mouse wheel to change the baseline coefficients and\nadjust the step using the $\\uparrow$ and $\\downarrow$ buttons.")
        ax.plot(ppm_scale[sl], S[sl], label='Experimental', lw=1.0, c='k')  # experimental
        ax.axvspan(true_limits[0], true_limits[1], color='y', alpha=0.2)
        ax.set_xlabel(r'$\delta$ (ppm)')
        ax.set_ylabel(r'Intensity (a.u.)')
        mathformat(ax)
        poly_plot, = ax.plot(ppm_scale[sl], y, label = 'Baseline', lw=0.8, c='tab:orange')

        # make pretty scale
        ax.set_xlim(max(limits[0],limits[1]),min(limits[0],limits[1]))
        pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
        pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)

        # Header for current values print
        head_print = ax.text(0.1, 0.04,
                '{:_^11}, {:_^11}, {:_^11}, {:_^11}, {:_^11}'.format(
                    'A', 'B', 'C', 'D', 'E'),
                ha='left', va='bottom', transform=fig.transFigure, fontsize=10, color='tab:orange')
        values_print = ax.text(0.1, 0.01,
                '{:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(
                    C[0], C[1], C[2], C[3], C[4]),
                ha='left', va='bottom', transform=fig.transFigure, fontsize=10)

        def update(val):
            # Calculates and draws all the figure elements
            y = polyn(x, C)

            # Update the plot
            poly_plot.set_ydata(y)
            values_print.set_text(
                    '{:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}, {:+5.2e}'.format(
                        C[0], C[1], C[2], C[3], C[4]))
            plt.draw()

        def reset(event):
            # Sets all the widgets to their starting values
            nonlocal C, om
            C = np.zeros(len(poly_name))
            om = np.zeros_like(C)
            update(0)       # to update the figure

        # Declare variables to store the final values
        C_f = np.zeros_like(C)
        def save(event):
            # Put current values in the final variables that are returned
            nonlocal C_f
            C_f = np.copy(C)


        # Connect widgets to functions
        poly_radio.on_clicked(statmod)
        up_button.on_clicked(switch_up)
        down_button.on_clicked(switch_down)
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
        save_button.on_clicked(save)
        reset_button.on_clicked(reset)

        ax.legend()
        plt.show()

        return C_f

    def write_input_INT(self, filename):
        """
        Write the input file for the fitting routine by calling a graphical interface.
        -------
        Parameters:
        - filename : str
            Name of the file to be written
        -------
        Returns:
        - matrix : 2darray
            Matrix with the values of the input file
        """
        #crea la matrice con nomi, ppm1, ppm2, v da zero in modo interattivo

        ppm_scale = self.pscale
        datasetp = self.datap

        limits, v_A, v_notA, grouped = self.select_regionsandpeaks()
        print(grouped)
        limiti_tot = []
        for i in range(len(v_A)):
            for j in range(len(limits)):
                if limits[j][1] <= v_A[i] <= limits[j][0]:
                    limiti_tot.append(limits[j])
                    break

        for i in range(len(v_notA)):
            for j in range(len(limits)):
                if limits[j][1] <= v_notA[i] <= limits[j][0]:
                    limiti_tot.append(limits[j])
                    break

        limiti_tot = np.array(limiti_tot)
        limiti_tot_o = limiti_tot[limiti_tot[:,0].argsort()]   #ordina dal più piccolo al più grande

        v_tot = v_A + v_notA
        v_tot.sort()

        names = []
        for i in range(len(v_tot)):
            if v_tot[i] in v_A:
                names.append('true')
            elif v_tot[i] in v_notA:
                names.append('false')

        v_tot = np.reshape(np.array(v_tot), (len(v_tot),))
        names = np.reshape(np.array(names), (len(names),))

        matrix = np.zeros((len(names), 4), dtype='object')
        matrix[:,0] = names
        matrix[:,1:3] = limiti_tot_o
        matrix[:,-1] = v_tot

        mult_numb = 0
        mult = np.zeros(matrix.shape[0], dtype='int64')
        for j in range(len(grouped)):
            for i in range(matrix.shape[0]):
                if matrix[i,-1] in grouped[j]:
                    mult[i] = j+1

        with open(filename, 'w') as f:
            f.write('name\tppm1\tppm2\tv\tmult\n')
            for i in range(len(matrix[:,0])):
                for j in range(len(matrix[0,:])):
                    if j==0:
                        f.write(matrix[i,j]+'\t')
                    else:
                        f.write('{:.5f}'.format(matrix[i,j])+'\t')
                f.write(str(mult[i]))
                f.write('\n')
            f.close()

        return matrix

    def int_guess_cycle(self, ns, ppm1, ppm2, taq, u, o1p, sf1, td, dw, cal=0, SI=0, SW=0, LB=0, SSB=0, completebsl=False):
        """
        Interactive guess for the fitting routine.
        -------
        Parameters:
        - ns : int
            Number of signals
        - ppm1 : float
            Left limit of the spectral region
        - ppm2 : float
            Right limit of the spectral region
        - taq : 1darray
            Acquisition time
        - u : 1darray
            Frequencies of the signals
        - o1p : float
            Carrier frequency
        - sf1 : float
            Spectrometer frequency
        - td : int
            Number of points
        - dw : float
            Dwell time
        - cal : float
            Calibration of ppmscale
        - SI : int
            Number of points in the spectrum
        - SW : float
            Spectral width
        - LB : float
            Line broadening
        - SSB : float
            squared-sine apodization factor
        - completebsl : bool
            If True, the baseline starts at the origin of the x ppmscale
        -------
        Returns:
        - k: 1darray
            Amplitudes of the signals
        - s: 1darray
            Standard deviations of the signals
        - phi: 1darray
            Phases of the signals
        - coeff: 1darray
            Coefficients of the baseline
        """

        ppmscale = self.pscale
        S = self.datap/np.max(self.datap.real)

        # Trim the spectrum and the scale at the given 'limits'
        lb = ppm1
        rb = ppm2
        lim1 = ppmfind(ppmscale, lb)[0]
        lim2 = ppmfind(ppmscale, rb)[0]
        xtrim, ytrim = trim_data(ppmscale, S, lb, rb)
        cal -= (ppmscale[0]-ppmscale[1])

        sx, dx, zero = find_limits(ppm1, ppm2, ppmscale)

        up_box = plt.axes([0.50, 0.92, 0.05, 0.02])
        down_box = plt.axes([0.50, 0.90, 0.05, 0.02])

        # Creation of sliders
        #   Creation of lists where to put boxes and sliders
        boxes = []
        sliders = []
        for i in range(ns):
            boxes.append([])
            sliders.append([])

        #   Configuration of slider values
        sl_labels = ['k', r'$\sigma$', r'$\phi$', r'$x_g$']
        radio_label = ['A', 'B', 'C', 'D', 'E']
        poli_coeff = np.zeros(len(radio_label))
        valmins = [-30, 0, -np.pi, 0]
        valmaxs = [10, np.abs(max(xtrim)-min(xtrim))/4, np.pi, 1]
        valinits = [-10, np.abs(max(xtrim)-min(xtrim))/8, 0, 0.2]
        orientations = ['horizontal', 'horizontal', 'horizontal', 'horizontal']

        for i in range(ns):
            # boxes[i].append(plt.axes([0.6, 0.95-i*0.1, 0.3, 0.02])) # k 0
            # boxes[i].append(plt.axes([0.6, 0.93-i*0.1, 0.3, 0.02])) # sigma 1
            # boxes[i].append(plt.axes([0.6, 0.91-i*0.1, 0.3, 0.02]))
            # boxes[i].append(plt.axes([0.6, 0.89-i*0.1, 0.3, 0.02]))
            boxes[i].append(plt.axes([0.6, 0.95-i*0.05, 0.3, 0.01])) # k 0  #x,y,larghezza,altezza
            boxes[i].append(plt.axes([0.6, 0.94-i*0.05, 0.3, 0.01])) # sigma 1
            boxes[i].append(plt.axes([0.6, 0.93-i*0.05, 0.3, 0.01]))
            boxes[i].append(plt.axes([0.6, 0.92-i*0.05, 0.3, 0.01]))

        radio_box = plt.axes([0.6, 0.02, 0.35, 0.08])
        reset_box = plt.axes([0.05, 0.90, 0.1, 0.04])
        save_box = plt.axes([0.2, 0.90, 0.1, 0.04])
        sande_box = plt.axes([0.35, 0.90, 0.1, 0.07])

        for i in range(ns):
            for j in range(len(sl_labels)):
                sliders[i].append(Slider(
                    ax=boxes[i][j],
                    label = sl_labels[j]+r'$_{'+str(i+1)+'}$',
                    valmin = valmins[j],
                    valmax = valmaxs[j],
                    valinit = valinits[j],
                    orientation = orientations[j]) )

        #   Make the buttons
        reset_button = Button(reset_box, 'RESET', hovercolor='0.975')
        save_button = Button(save_box, 'SAVE', hovercolor='0.975')
        saveandexit = Button(sande_box, 'SAVE\nAND\nEXIT', hovercolor='0.975')
        up_button = Button(up_box, '$\\uparrow$', hovercolor='0.975')
        down_button = Button(down_box, r'$\downarrow$', hovercolor='0.975')
        radio_radio = RadioButtons(radio_box, radio_label)

        om = np.zeros(len(radio_label))
        stats = np.zeros_like(om)
        stats[0] = 1

        def statmod(label):
            nonlocal stats
            stats = np.zeros_like(om)
            for k, L in enumerate(radio_label):
                if label == L:
                    stats[k]=1

        def roll_up(event):
            nonlocal poli_coeff
            for k in range(len(radio_label)):
                if stats[k]:
                    poli_coeff[k]+=10**om[k]

        def roll_down(event):
            nonlocal poli_coeff
            for k in range(len(radio_label)):
                if stats[k]:
                    poli_coeff[k]-=10**om[k]

        def up_om(event):
            nonlocal om
            for k in range(len(radio_label)):
                if stats[k]:
                    om[k] += 1

        def down_om(event):
            nonlocal om
            for k in range(len(radio_label)):
                if stats[k]:
                    om[k] -= 1

        def poli(x, poli_coeff):
            p = np.zeros_like(x)
            for i in range(len(poli_coeff)):
                p+=poli_coeff[i]*x**i
            return p

        def on_scroll(event):
            nonlocal y
            if event.button == 'up':
                roll_up(event)
            elif event.button == 'down':
                roll_down(event)

            y = poli(x, poli_coeff)
            update(0)

        def update(val):

            PV_inside = np.zeros_like(PV)
            for i in range(ns):

                k = sliders[i][0].val
                s = sliders[i][1].val
                phi = sliders[i][2].val
                xg = sliders[i][3].val

                lor = t_voigt(taq, (u[i]-o1p+cal)*sf1, s*2*np.pi*sf1, A = 2**k, phi=phi, x_g=xg)
                ### processing
                lor *= em(lor, LB, SW)
                lor *= qsin(lor, SSB)
                lor = zf(lor, SI)
                ###
                pv[i] = ft(lor, SI, dw, o1p, sf1)[0]
                pv[i] = np.conj(pv[i])[::-1]
                PV_inside += pv[i][sx:dx].real

           
            Cost = np.max(S.real)#np.max(S[sx:dx].real)#np.sum((PV_inside+y)*S[sx:dx].real)/np.sum((PV_inside+y)**2)


            for i in range(ns):
                gaussian[i].set_ydata(Cost*pv[i][sx:dx].real)

            polinomio.set_ydata(Cost*y)

            totale.set_ydata((Cost*(y+PV_inside)))
            #print(om,end='\r')
            plt.draw()

        def reset(event):
            nonlocal poli_coeff, om
            # Function for the RESET button:
            # Resets the slider values to the initial condition
            for i in range(ns):
                for j in range(len(sl_labels)):
                    sliders[i][j].reset()
            poli_coeff = np.zeros(len(radio_label))
            om = np.zeros_like(poli_coeff)
            update(0)

        if completebsl:
            zero=0
        else:
            pass

        x = ppmscale[sx:dx]-zero
        y = np.zeros_like(x)

        k = []
        fwhm = []
        phi = []
        xg = []
        def save(event):
            # Function for the SAVE button:
            # Writes the current parameters in the 'filename' file
            nonlocal k, fwhm, phi, xg
            k = []
            fwhm = []
            phi = []
            xg = []
            for i in range(ns):
                k.append(sliders[i][0].val)
                fwhm.append(sliders[i][1].val)
                phi.append(sliders[i][2].val)
                xg.append(sliders[i][3].val)


        def save_and_exit(event):
            # Function for the SAVE AND EXIT button:
            # Calls the 'save' function, then closes the figure
            save(event)
            plt.close()


        # 'pv' stores the single pseudo-voigts, PV is the sum of all of them (i.e. the real fitting function)
        pv = []
        PV = np.zeros_like(ppmscale[sx:dx])
        for i in range(ns):
            pv.append(np.zeros_like(ppmscale))
        # plot iniziale
        gaussian = []   # Here the plots of 'pv's are stored, in order to be updated

        for i in range(ns):

            lor = t_voigt(taq, (u[i]-o1p+cal)*sf1, np.abs(max(xtrim)-min(xtrim))/8*2*np.pi*sf1, A=2**(-10), phi = 0, x_g = 0.2)

            ### processing
            lor *= em(lor, LB, SW)
            lor *= qsin(lor, SSB)
            lor = zf(lor, SI)
            ###

            pv[i] = ft(lor, SI, dw, o1p, sf1)[0]
            pv[i] = np.conj(pv[i])[::-1]
            PV += pv[i][sx:dx].real

        Cost = np.max(S.real)#np.sum((PV+y)*S[sx:dx].real)/np.sum((PV+y)**2)

        # Creation of the interactive figure panel
        fig = plt.figure(1)
        fig.set_size_inches(13,10)
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.55)     # Make room for the sliders
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r'$\delta$ (ppm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_xlim(lb, rb)
        ax.plot(ppmscale, S, c='b', lw=0.8, label='Spectrum')  # Plots the original data

        for i in range(ns):
            temp, = ax.plot(ppmscale[sx:dx], Cost*pv[i][sx:dx].real, lw=0.8, label='Signal '+str(i+1))      # Plot of 'pv'
            gaussian.append(temp)
        polinomio, = ax.plot(x+zero, Cost*y, lw=0.8, label='Baseline')
        totale, = ax.plot(ppmscale[sx:dx], Cost*(y+PV),lw=0.9,c='r', label='TOTAL')

        # Call the 'update' functions upon interaction with the widgets
        #   Sliders
        for i in range(ns):
            for j in range(len(sl_labels)):
                sliders[i][j].on_changed(update)

        #   Buttons
        save_button.on_clicked(save)
        reset_button.on_clicked(reset)
        saveandexit.on_clicked(save_and_exit)
        up_button.on_clicked(up_om)
        down_button.on_clicked(down_om)
        radio_radio.on_clicked(statmod)
        scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

        ax.legend()

        plt.show()
        plt.close()

        return 2**np.array(k),fwhm, phi, xg, poli_coeff[0], poli_coeff[1], poli_coeff[2], poli_coeff[3], poli_coeff[4]

    def interactive_iguess(self, matrix, filename, acqupars, procpars, completebsl=False):
        """
        Interactive generation of initial guess for fitting of a 1D NMR spectrum with 'ns' Lorentzian signals.
        Returns the location of the generated file, 'filename'.
        -------
        Parameters:
        - matrix : 2darray
            Matrix with the values of the input file
        - filename : str
            Name of the file to be written
        - acqupars : dict
            Acquisition parameters
        - procpars : dict
            Processing parameters
        - completebsl : bool
            If True, the baseline starts at the origin of the x ppmscale
        -------
        Returns:
        - filename : str
            Name of the file where the parameters have been written
        """
        # Interactive generation of initial guess for fitting of a 1D NMR spectrum with 'ns' Lorentzian signals.
        # Returns the location of the generated file, 'filename'.

        ppm1 = matrix[:,1]
        ppm2 = matrix[:,2]
        ppm1_set = list(set(ppm1))
        ppm2_set = list(set(ppm2))
        ppm1_set.sort()
        ppm2_set.sort()

        DE = acqupars['DE'] 
        SR = procpars['SR'] 
        SI = procpars['SI'] 
        SW = acqupars['SW'] 
        LB = procpars['LB']
        SSB = procpars['SSB']
        TD = acqupars['TD']
        o1 = acqupars['o1']
        sf1 = acqupars['SFO1']
        
        dw = 1/(SW)
        td = TD//2
        o1p = o1/sf1
        taq = np.linspace(0, td*dw, td) + DE
        cal2 = SR/sf1

        for it in range(len(ppm1_set)):

            ppm1 = ppm1_set[it]
            matrix_short = []
            for j in range(len(matrix[:,0])):
                if ppm1 == matrix[j,1]:
                    matrix_short.append(matrix[j,:])

            matrix_short = np.array(matrix_short)
            u = matrix_short[:,-1]
            ns = len(matrix_short[:,0])

            k,fwhm,phi,xg, A, B, C, D, E = self.int_guess_cycle(ns, ppm1_set[it], ppm2_set[it], taq, u, o1p, sf1, td, dw, cal2, SI, SW, LB, SSB, completebsl)
            self.save_pvpar(filename, k,fwhm,phi,xg, A,B,C,D,E, it, ppm1_set[it], ppm2_set[it])

    def save_pvpar(self, filename, k, fwhm,phi,xg, A, B, C, D, E, it, sx, dx):
        # Saves the parameters of the 'ns' Lorentzian functions in a file named 'filename'
        ns = len(fwhm)
        if it == 0:
            f = open(filename, 'w')
            f.write('i\tppm1\tppm2\tk\t\tfwhm\t\tphi\t\txg\t\tA\t\tB\t\tC\t\tD\t\tE\n')
        else:
            f = open(filename, 'a')
        for i in range(ns):
            f.write(str(i)+'\t'+
                    '{:.1f}'.format(sx)+'\t'+
                    '{:.1f}'.format(dx)+'\t'+
                    '{:.5e}'.format(k[i])+'\t'+
                    '{:.5e}'.format(fwhm[i])+'\t'+
                    '{:.5e}'.format(phi[i])+'\t'+
                    '{:.5e}'.format(xg[i])+'\t'+
                    '{:.3e}'.format(A)+'\t'+
                    '{:.3e}'.format(B)+'\t'+
                    '{:.3e}'.format(C)+'\t'+
                    '{:.3e}'.format(D)+'\t'+
                    '{:.3e}'.format(E)+'\n')
        f.close()

def read_input_INT(filename1):
    #legge nomi, ppm1, ppm2, v da file
    with open(filename1) as f:
        file = f.readlines()

    matrix = np.zeros((len(file)-1, 5), dtype='object')
    for i,line in enumerate(file):
        if i > 0:
            splitline = line.split('\t')
            try:
                splitline.remove('\n')
            except:
                pass
            for j in range(len(splitline)):
                # print(splitline)
                try:
                    elem = float(splitline[j])
                    matrix[i-1,j] = elem
                except:
                    elem = str(splitline[j])
                    matrix[i-1,j] = elem

    return matrix

def read_input_INT2(filename2, matrix):

    with open(filename2, 'r') as f:
        F = f.readlines()
    tensore = np.zeros((len(matrix[:,0]), len(matrix[0,:])+9), dtype='object')
    splitline = []
    for i,line in enumerate(F):
        if i!=0:
            splitline.append(line.split('\t'))
    splitline = np.array(splitline, dtype = 'float64')

    for j in range(len(tensore[:,0])):
        for k in range(8):
            tensore[j, 4+k] = float(splitline[j,3+k])
    tensore[:,:4] = matrix

    return tensore

def nmr_spectra_1d(path):

    if '/pdata/' in path: 
        pass
    else:
        path = path + '/pdata/1'

    dic, datare = ng.bruker.read_pdata(path, bin_files=['1r'])   
    dic, dataim = ng.bruker.read_pdata(path, bin_files=['1i'])   
    datap = datare + 1j*dataim      

    udic = ng.bruker.guess_udic(dic, datap)
    C = ng.convert.converter()
    C.from_bruker(dic, datap, udic)
    dicpipe, datapipe = C.to_pipe()
    uc = ng.pipe.make_uc(dicpipe, datapipe)
    ppm_scale = uc.ppm_scale()

    return datap, ppm_scale, dic

def nmr_spectra_pseudo2d(path):

    dic, datare = ng.bruker.read_pdata(path, bin_files=['2rr'])   #read only the real data dic:matrioska di dizionari che racchiude tutti i parametri che ci sono nella cartella dello spettro
    #dic, dataim = ng.bruker.read_pdata(path, bin_files=['2ii'])   #read only the imaginary data

    #dataim = np.reshape(dataim, datare.shape)

    data = datare + 1j*np.zeros_like(datare)      #recombine Re and Im to get the complete complex data

    udic = ng.bruker.guess_udic(dic, data)
    C = ng.convert.converter()
    C.from_bruker(dic, data, udic)
    dicpipe, datapipe = C.to_pipe()
    uc = ng.pipe.make_uc(dicpipe, datapipe)
    ppm_scale = uc.ppm_scale()

    return data, ppm_scale, dic
    
def histogram(data, nbins=100, density=True, f_lims= None, xlabel=None, x_symm=False, name=None, ret_stats=False):
    """
    Computes an histogram of 'data' and tries to fit it with a gaussian lineshape.
    The parameters of the gaussian function are calculated analytically directly from 'data'
    --------
    Parameters:
    - data : ndarray
        the data to be binned
    - nbins : int
        number of bins to be calculated
    - density : bool
        True for normalize data
    - f_lims : tuple or None
        limits for the x axis of the figure
    - xlabel : str or None
        Text to be displayed under the x axis
    - x_symm : bool
        set it to True to make symmetric x-axis with respect to 0
    - name : str
        name for the figure to be saved
    - ret_stats : bool
        if True, returns the mean and standard deviation of data.
    -------
    Returns:
    - m : float
        Mean of data
    - s : float
        Standard deviation of data.
    """

    from scipy import stats

    if len(data.shape) > 1:
        data = data.flatten()

    if x_symm:
        lims = (- max(np.abs(data)), max(np.abs(data)) )
    else:
        lims = (min(data), max(data))

    hist, bin_edges = np.histogram(data, bins=nbins, range=lims, density=density)   # Computes the bins for the histogram

    lnspc = np.linspace(lims[0], lims[1], len(data))        # Scale for a smooth gaussian
    m, s = stats.norm.fit(data)                                 # Get mean and standard deviation of 'data'

    if density:
        A = 1
    else:
        A = np.trapz(hist, dx=bin_edges[1]-bin_edges[0])    # Integral
    fit_g = A / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((lnspc - m) / s)**2) # Gaussian lineshape

    fig = plt.figure()
    fig.set_size_inches(4.96, 3.78)
    ax = fig.add_subplot(1,1,1)
    ax.hist(data, color='tab:blue', density=density, bins=bin_edges)
    ax.plot(lnspc, fit_g, c='r', lw=0.6, label = r'$\mu = ${:.3g}'.format(m)+'\n'+r'$\sigma = ${:.3g}'.format(s))
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis='both', style='scientific', scilimits=(-3,3), useMathText=True)
    ax.yaxis.get_offset_text().set_size(7)
    ax.xaxis.get_offset_text().set_size(7)
    if density:
        ax.set_ylabel('Normalized count', fontsize=8)
    else:
        ax.set_ylabel('Count', fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if f_lims:
        ax.set_xlim(f_lims)
    ax.legend(loc='upper right', fontsize=6)
    fig.tight_layout()
    if name:
        plt.savefig(name+'.png', format='png', dpi=600)
    else:
        plt.show()
    plt.close()

    if ret_stats:
        return m, s
        
def f_figure_comp(ppmscale, data, model, comp, name=None, basefig=None, dic_fig={'h':3.59,'w':2.56,'sx':None,'dx':None}):
	
    fig = plt.figure()
    fig.set_size_inches(dic_fig['h'],dic_fig['w'])   #3.59,2.56
    plt.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.90)
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(labelsize=6.5)
    if basefig is not None:
        ax.plot(ppmscale, basefig, 'r',lw=0.5, label='baseline')
    ax.plot(ppmscale, data, lw=0.5, label='experiment')
    ax.plot(ppmscale, model, lw=0.5, label='model')
    ax.plot(ppmscale, data-model, lw=0.5, label='residue')
    for i in range(len(comp)):
        ax.plot(ppmscale, comp[i], '--', lw=0.4, label='comp. '+str(i+1))
    ax.set_xlabel(r'$\delta$ (ppm)', fontsize=8)
    ax.set_ylabel('Intensity (a.u.)', fontsize=8)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
    ax.yaxis.get_offset_text().set_size(7)
    if dic_fig['sx'] and dic_fig['dx']:
        ax.set_xlim(dic_fig['dx'],dic_fig['sx'])
    ax.invert_xaxis()
    ax.legend(fontsize=6)
    if name is None:
        plt.show()
    else:
        plt.savefig(name+'.png', dpi=600)
        plt.close()	

def fig_stacked_plot(ppmscale, data, baseline, delays_list, limits, lines, name=None, map='rainbow', dic_fig={'h':5,'w':4,'sx':None,'dx':None}, f_legend=False):

    #colors from map
    cmap = plt.get_cmap(map)
    colors = cmap(np.linspace(0, 1, len(delays_list)))

    if dic_fig['sx'] and dic_fig['dx']:
        sx,dx,_ = find_limits(dic_fig['sx'],dic_fig['dx'],ppmscale)
    else:
        sx,dx,_ = find_limits(limits[0],limits[1],ppmscale)

    fig = plt.figure()
    fig.set_size_inches(dic_fig['h'],dic_fig['w'])   
    plt.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.90)
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(labelsize=6.5)
    for i in range(len(delays_list)):
        if f_legend:
            ax.plot(ppmscale[sx:dx], data[i,sx:dx]-baseline[sx:dx], lw=0.3,c=colors[i], label=f'D: {delays_list[i]:.2e}')
        else:
            ax.plot(ppmscale[sx:dx], data[i,sx:dx]-baseline[sx:dx], lw=0.3,c=colors[i])
    if lines:
        for i in range(len(lines)):
            ax.axvline(lines[i], c=colors[i], lw=0.3)
    ax.set_xlabel(r'$\delta$ (ppm)', fontsize=8)
    ax.set_ylabel('Intensity (a.u.)', fontsize=8)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
    ax.yaxis.get_offset_text().set_size(7)
    if dic_fig['sx'] and dic_fig['dx']:
        ax.set_xlim(dic_fig['dx'],dic_fig['sx'])
    else:
        ax.set_xlim(limits[0],limits[1])
    ax.invert_xaxis()
    if f_legend:
        ax.legend(fontsize=5)
    if name is None:
        plt.show()
    else:
        plt.savefig(name+'.png', dpi=600)
        plt.close()



##############
class color_term:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
##############
