
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
github repository: letiziafiorucci/RELAXID
Last update: 25/09/2024

!!! DISCLAIMER !!!
All functions for spectra processing and handling are also availbale in KLASSEZ.
(github repository: MetallerTM/KLASSEZ)

This file contains a collection of functions for the extraction of various types of fit parameters, 
i.e. heights, intensity, integral, shift and linewidth, either from: 
- a single 1D spectrum, 
- a series of 1D spectra (treated as a pseudo2D),
- a single pseudo2D spectrum, 
- a series of pseudo2D spectra.

This file contains one example for each of the possible use of this collction of functions, i.e., in this order:
1. intensity_fit_pseudo2D: fit of the intensity of a series of pseudo2D spectra,
2. intensity_fit_1D: fit of the intensity of a series of 1D spectra treated as a pseudo2D,
3. model_fit_1D: fit of a series of 1D spectra through peak modeling treated as a pseudo2D,
4. model_fit_pseudo2D: fit of a series of pseudo2D spectra through peak modeling.
5. model_fit_1D_reg: fit of a series of 1D spectra through peak modeling treated as a pseudo2D with regularization.

'''

from f_fit import *

#mettere i limiti di integrazione definiti a mano invece che con l'intefaccia, stessa cosa per i coefficienti della baseline

### FUNCTION 1 ###
if 0: #test passed
    # test for intensity fit for series of pseudo2D 

    # folder containing the spectra
    path = '/home/letizia/Documents/Dottorato/Programma_new/relax/data600/dati_olio600'

    print('MAIN SPECTRA PATH:' )
    print(path)
    
    list_path = [int(f) for f in os.listdir(path) if not f.startswith('.')]
    list_path = [str(f) for f in np.sort(list_path)]

    # list of delays
    delays_list = [np.loadtxt(path+'/'+list_path[i]+'/vdlist')+3e-3 for i in range(len(list_path))]

    # intensity fit
    dir_result = intensity_fit_pseudo2D(
                                        path, # path of the main spectra
                                        delays_list, # list of delays (or any x value)
                                        list_path, # list of folders
                                        prev_lims = True, # if True, the same region of integration is used for all the spectra
                                        prev_coeff = True, # if True, the same coefficients for the baseline are used for all the spectra
                                        area=True, # if True, the integral of selected regions is considered, otherwise the intensity of the highest point is considered
                                        VCLIST=None, # list of fields (can also be just an array of consecutive numbers)
                                        cal_lim = (1.2,0.888), # limits for the calibration
                                        baseline=True, # if True, the baseline is subtracted
                                        doexp=True, # if True, the intensities are fitted with an exponential decay
                                        f_int_fit=None, # function for the intensity fit
                                        fargs=None # arguments of the intensity fit, except for x and y
                                        )
    # the limits of integration are ordered from the smallest ppmvalue to the largest (as also stated in the output file)
    theUltimatePlot(dir_result, list_path, bi_list=[], colormap = 'hsv', area=True)
    
### FUNCTION 2 ###
if 0: #test passed
    # test for series of 1D spectra to be treated as pseudo2D

    # folder containing the spectra
    path = '/home/letizia/Documents/NMR/19FNMR_Luchinat_27spectra/' 

    print('MAIN SPECTRA PATH:' )
    print(path)

    num_sp = list(np.arange(101,128,1))  
    list_sp = [str(i)+'/pdata/1' for i in num_sp]

    delays = np.linspace(0, 1, len(list_sp)) #start, stop, n. steps

    dir_result = intensity_fit_1D(
                                path, # path of the main spectra
                                delays, # list of delays (or any x value)
                                list_sp, # list of folders
                                area=True, # if True, the integral of selected regions is considered, otherwise the intensity of the highest point is considered
                                auto_ph = False, # if True, the phase is automatically adjusted  #not working at the moment
                                cal_lim = (-72.52,-72.94), # limits for the calibration
                                baseline=True, # if True, the baseline is subtracted
                                delta = 2, # expansion of the baseline correction region
                                doexp=False, # if True, the intensities are fitted with an exponential decay
                                f_int_fit=None, # function for the intensity fit
                                fargs=None # arguments of the intensity fit, except for x and y
                                )

### FUNCTION 3 ###
if 1:
    # test for series of 1D spectra to be treated as pseudo2D

    # folder containing the spectra
    path = '/home/letizia/Documents/NMR/19FNMR_Luchinat_27spectra/'

    print('MAIN SPECTRA PATH:' )
    print(path) 

    # in this case the experiment names correspond to consecutive numbers (from 27 to 43)
    # in other cases the experiment names can be read from the folder or directly in the code as:
    # list_sp = ['spectrum1/pdata/1', 'spectrum2/pdata/1', 'spectrum3/pdata/1']
    num_sp = list(np.arange(101,128,1))   
    list_sp = [str(i)+'/pdata/1' for i in num_sp]

    # in this case I read the delays from the title of the spectra
    # this is just an example, the delays can be read from a file or defined directly in the code as:
    # delays = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    delays = np.linspace(0, 1, len(list_sp)) 

    # limits for the fit parameters defined in absolute terms as 'key':(min,max)
    # the shift is the only parameter whose limits are defined in terms of a delta value to be applied to the initial guess
    # key can be 'shift', 'k', 'lw', 'ph', 'xg', 'A', 'B', 'C', 'D', 'E'
    # if the max and min are defined equal, the parameter is fixed
    # if the limit is not defined, the parameter is free to vary between +np.inf and -np.inf
    lim1 = {'shift':(-1,1), 'lw':(1e-4,2.5), 'ph':(-np.pi/20,np.pi/20),'A':(0,0), 'B':(0,0), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
    # the second limits are defined not in absolute terms but as a percentage of the initial value
    # this second dict will be used for the spectra subsequent to the first one (the one used in the definition of the initial guess)
    # e.g. if shift = (0.9, 1.1) means that the value of the shift is allowed to vary between 0.9*initial and 1.1*initial
    lim2 = {'shift':(0.9,1.1), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)} 

    # if lim1 is not defined, then the default limits are used:
    # lim1 = {'shift':(-2,2), 'k':(0,1), 'lw':(1e-4,3.5), 'ph':(-np.pi,np.pi), 'xg':(0,1)} 
    # if lim2 is not defined, the same limits as lim1 are used also for the subsequent spectra

    model_fit_1D(
                path, # path of the main spectra
                delays, # list of delays (or any x value)
                list_sp, # list of folders
                cal_lim = None, # limits for the calibration
                dofit=True, # if True, the spectra are fitted
                prev_fit = None, # directory of the previous fit (if any)
                fast = True, # if True, the fitting is performed with a faster algorithm
                limits1 = lim1, # limits for the fit parameters of the first spectrum
                limits2 = lim2,  # limits for the fit parameters of the subsequent spectra
                L1R = None, # factor for L1 regularization (Lasso)
                L2R = None, # factor for L2 regularization (Ridge) 
                doexp=False, # if True, the intensities are fitted with an exponential decay
                f_int_fit=None, # function for the intensity fit
                fargs=None, # arguments of the intensity fit, except for x and y
                ) 

### FUNCTION 4 ###
if 0: 
    # test for series of pseudo2D 

    ### maybe this demonstrates that I also need the other term in calibration

    path = '/home/letizia/Documents/NMR/PPyr0.2mM_MMP12_4uM_TRIS_12_01_23/'

    print('MAIN SPECTRA PATH:' )
    print(path)

    list_path = [int(f) for f in os.listdir(path) if not f.startswith('.')]
    list_path = [str(f) for f in np.sort(list_path)]
    delays_list = [np.loadtxt(path+list_path[i]+'/vdlist')+3e-3 for i in range(len(list_path))]

    # limits for the fit parameters defined in absolute terms as 'key':(min,max)
    # the shift is the only parameter whose limits are defined in terms of a delta value to be applied to the initial guess
    # key can be 'shift', 'k', 'lw', 'ph', 'xg', 'A', 'B', 'C', 'D', 'E'
    # if the max and min are defined equal, the parameter is fixed
    # if the limit is not defined, the parameter is free to vary between +np.inf and -np.inf
    lim1 = {'shift':(-0.05,0.05), 'lw':(1e-4,0.05), 'ph':(-np.pi/20,np.pi/20), 'k':(0,1), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
    # the second limits are defined not in absolute terms but as a percentage of the initial value
    # this second dict will be used for the spectra subsequent to the first one (the one used in the definition of the initial guess)
    # e.g. if shift = (0.9, 1.1) means that the value of the shift is allowed to vary between 0.9*initial and 1.1*initial
    lim2 = {'shift':(0.95,1.05), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)} 

    # if lim1 is not defined, then the default limits are used:
    # lim1 = {'shift':(-2,2), 'k':(0,1), 'lw':(1e-4,3.5), 'ph':(-np.pi,np.pi), 'xg':(0,1)} 
    # if lim2 is not defined, the same limits as lim1 are used also for the subsequent spectra

    dir_result = model_fit_pseudo2D(
                path, # path of the main spectra
                delays_list, # list of delays (or any x value)
                list_path, # list of folders
                cal_lim = (1.90,1.70), # limits for the calibration
                fast = True, # if True, the fitting is performed with a faster algorithm
                dofit = True, # if True, the spectra are fitted
                prev_fit = None, #'PPyr0.2mM_MMP12_4uM_TRIS_12_01_23_modelfit_ex/',
                limits1 = lim1, # limits for the fit parameters of the first spectrum
                limits2 = lim2,  # limits for the fit parameters of the subsequent spectra
                prev_guess = True, # if True, the previous fit is used as initial guess without asking
                L1R = 8, # factor for L1 regularization (Lasso)
                L2R = None, # factor for L2 regularization (Ridge) 
                doexp=True, # if True, the intensities are fitted with an exponential decay
                f_int_fit=None, # function for the intensity fit
                fargs=None, # arguments of the intensity fit, except for x and y
                )
    
    # the limits of integration are ordered from the smallest ppmvalue to the largest (as also stated in the output file)
    theUltimatePlot('PPyr0.2mM_MMP12_4uM_TRIS_12_01_23_modelfit/', list_path, bi_list=[], colormap = 'hsv', area=True)
    
### FUNCTION 5 ###
if 0:
    # test for series of 1D spectra to be treated as pseudo2D

    # folder containing the spectra
    path = '/home/letizia/Documents/NMR/FDX2_13C15N_700b_280224/'

    print('MAIN SPECTRA PATH:' )
    print(path) 

    # in this case the experiment names correspond to consecutive numbers (from 27 to 43)
    # in other cases the experiment names can be read from the folder or directly in the code as:
    # list_sp = ['spectrum1/pdata/1', 'spectrum2/pdata/1', 'spectrum3/pdata/1']
    num_sp = list(np.arange(27,44,1))   #pochescan 
    list_sp = [str(i)+'/pdata/1' for i in num_sp]

    # in this case I read the delays from the title of the spectra
    # this is just an example, the delays can be read from a file or defined directly in the code as:
    delays = []
    for idx in range(len(list_sp)):
        title = open(path+list_sp[idx]+'/title').readlines()
        title = title[0].split(' ')
        delays.append(float(title[-1][:title[-1].index('m')])*1e-3)

    # limits for the fit parameters defined in absolute terms as 'key':(min,max)
    # the shift is the only parameter whose limits are defined in terms of a delta value to be applied to the initial guess
    # key can be 'shift', 'k', 'lw', 'ph', 'xg', 'A', 'B', 'C', 'D', 'E'
    # if the max and min are defined equal, the parameter is fixed
    # if the limit is not defined, the parameter is free to vary between +np.inf and -np.inf
    lim1 = {'shift':(-1,1), 'lw':(1e-4,2.5), 'ph':(-np.pi/20,np.pi/20), 'B':(0,0), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
    # the second limits are defined not in absolute terms but as a percentage of the initial value
    # this second dict will be used for the spectra subsequent to the first one (the one used in the definition of the initial guess)
    # e.g. if shift = (0.9, 1.1) means that the value of the shift is allowed to vary between 0.9*initial and 1.1*initial
    lim2 = {'shift':(0.95,1.05), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)} 

    # if lim1 is not defined, then the default limits are used:
    # lim1 = {'shift':(-2,2), 'k':(0,1), 'lw':(1e-4,3.5), 'ph':(-np.pi,np.pi), 'xg':(0,1)} 
    # if lim2 is not defined, the same limits as lim1 are used also for the subsequent spectra

    model_fit_1D_reg(
                path, # path of the main spectra
                delays, # list of delays (or any x value)
                list_sp, # list of folders
                cal_lim = None, # limits for the calibration
                dofit=True, # if True, the spectra are fitted
                limits1 = lim1, # limits for the fit parameters of the first spectrum
                limits2 = lim2,  # limits for the fit parameters of the subsequent spectra
                L1R = None, # factor for L1 regularization (Lasso)
                L2R = 200, # factor for L2 regularization (Ridge) !!!!very helpful for big baseline distortions
                doexp=False, # if True, the intensities are fitted with an exponential decay
                f_int_fit=None, # function for the intensity fit
                fargs=None, # arguments of the intensity fit, except for x and y
                ) 
