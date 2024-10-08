
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
github repository: letiziafiorucci/RELAXID
Last update: 30/09/2024

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

'''

from f_fit import *

#mettere i limiti di integrazione definiti a mano invece che con l'intefaccia, stessa cosa per i coefficienti della baseline
    
### FUNCTION 1 ###
if 0: #test passed
    # test for series of 1D spectra to be treated as pseudo2D

    # folder containing the spectra
    path = 'example1' 

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
    
### FUNCTION 2 ###
if 0: #test passed
    # test for intensity fit for series of pseudo2D 

    # folder containing the spectra
    path = 'example2'

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

### FUNCTION 3 ###
if 0:
    # test for series of 1D spectra to be treated as pseudo2D

    # folder containing the spectra
    path = 'example3'

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
    theUltimatePlot(dir_result, list_path, bi_list=[], colormap = 'hsv', area=True) 



if 0:

    import klassez as kl


    path = '/home/letizia/Documents/Dottorato/Fit_tools/test/Q/DMTP-1_PROTON_nt1_28C_01.fid' 

    S = kl.Spectrum_1D(path, spect="varian")    
    S.process()
    sr = S.S
    ppm_scale = S.ppm

    pprint(S.procs)

    acqupars = {
                'DE':0,
                'SW':S.acqus['SW'],
                'TD':S.acqus['TD']//2,
                'o1':S.acqus['o1'],
                'SFO1':S.acqus['SFO1']
                }
    
    procpars = {
                'SR':0,
                'SI':len(sr),
                'LB':S.procs['wf']['lb'],
                'SSB':S.procs['wf']['ssb']
                }

    print('MAIN SPECTRA PATH:' )
    print(path)
 
    list_sp = ['']

    delays = [0.0] #start, stop, n. steps

    lim1 = {'shift':(-1,1), 'lw':(1e-4,2.5), 'ph':(0,0),'A':(0,0), 'B':(0,0), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
    lim2 = {'shift':(0.9,1.1), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)} 

    model_fit_1D(
                path, # path of the main spectra
                delays, # list of delays (or any x value)
                list_sp, # list of folders
                cal_lim = None, # limits for the calibration
                dofit=True, # if True, the spectra are fitted
                prev_fit = None, # directory of the previous fit (if any)
                fast = False, # if True, the fitting is performed with a faster algorithm
                limits1 = lim1, # limits for the fit parameters of the first spectrum
                limits2 = lim2,  # limits for the fit parameters of the subsequent spectra
                L1R = None, # factor for L1 regularization (Lasso)
                L2R = None, # factor for L2 regularization (Ridge) 
                doexp=False, # if True, the intensities are fitted with an exponential decay
                f_int_fit=None, # function for the intensity fit
                fargs=None, # arguments of the intensity fit, except for x and y
                Spectra=np.array([sr]), 
                ppmscale=ppm_scale, 
                acqupars=acqupars,
                procpars=procpars
                )
    

if 1:

    import scipy

    def calc_shift(temp, S1, S2, A_h, J, gammaC=267/4):

        def energy(Si, J):
            return 1/2*J*(Si*(Si+1))

        #gammaC = 267/4   #672828/1e6  # 2piMHz/T  #-267/10
        conv = 1.9865e-23 #lo moltiplico per cm-1 per ottenere J
        kB = scipy.constants.k
        muB = scipy.constants.physical_constants['Bohr magneton'][0]
        ge = 2.0023
        pref = 2*np.pi*ge*muB/(3*kB*gammaC*temp)

        sum1 = 0
        sum2 = 0
        for s in np.arange(np.abs(S1-S2), S1+S2+1, 1):
            sum1 += 1/2*s*(s+1)*(2*s+1)*np.exp(-energy(s, J*conv)/(kB*temp))
            sum2 += (2*s+1)*np.exp(-energy(s, J*conv)/(kB*temp))

        sum1 *= pref*A_h*1e6
        shift = sum1/sum2

        return shift


    def T_model(param, temp, shift, result=False, T_long=None):
        
        S = 5/2
        par = param.valuesdict()
        J = par['J']
        A = [par['A_'+str(i)] for i in range(shift.shape[1])]
        res = []
        conshift = np.zeros_like(shift)
        for i in range(shift.shape[0]):  #per temp
            for j in range(shift.shape[1]):  #per protone
                conshift_s = calc_shift(temp[i], S, S, A[j], J)#[j])
                conshift[i,j] = conshift_s
                res.append(conshift_s-shift[i,j])

        if result:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(shift.shape[1]):
                line, = ax.plot(1000/temp, shift[:,i], 'o')
                ax.plot(1000/temp, conshift[:,i], '--', c=line.get_color())
            plt.show()
            plt.close()

            cont_list = []
            for j in range(shift.shape[1]):
                cont_list.append([calc_shift(t, S, S, A[j], J) for t in T_long])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(shift.shape[1]):
                line, = ax.plot(1000/temp, shift[:,i], 'o', label=f'{i+1}: J = {J:.1f} cm-1; A/h = {A[i]:.3f} MHz')
                ax.plot(1000/temp, conshift[:,i], '--', c=line.get_color())
                ax.plot(1000/T_long, cont_list[i], lw = 0.7, c='k')
            plt.legend()
            plt.xlabel('1000/T (K-1)')
            plt.ylabel('Shift (ppm)')
            plt.show()
            plt.close()
            return conshift
        else:
            return res


    def cal_temp(T):
        #temperature calibration at 1200 MHz

        T_cal = []
        for t in T:
            T_cal.append(t*0.955944+9.81982517+2)

        return T_cal


    #--------------------------------------#
    # Temp shift dep FDX2_13C15N_1200_090224
    #--------------------------------------#
    path = '/home/letizia/Documents/NMR/FDX2_13C15N_1200_090224_/'
    num_sp = list(np.arange(23,31,1))   #pochescan
    list_sp = [str(i)+'/pdata/10' for i in num_sp]

    temp = []
    for idx in range(len(list_sp)):
        title = open(path+list_sp[idx]+'/title').readlines()
        title = title[0].split(' ')
        temp.append(float(title[2][:title[2].index('K')]))

    temp = cal_temp(temp)  #calibration of the temperature at 1200 MHz

    lim1 = {'shift':(-1,1), 'lw':(1e-4,2.5), 'ph':(-np.pi/20,np.pi/20)}
    lim2 = {'lw':(0,0), 'ph':(0,0), 'xg':(0,0), 'A':(0.9,1.1), 'B':(0.9,1.1), 'C':(0.9,1.1), 'D':(0.9,1.1), 'E':(0.9,1.1)} 


    _, shift_tot = model_fit_1D(
                                path, 
                                temp, 
                                list_sp, 
                                cal_lim = None, 
                                dofit=True, 
                                prev_fit = None, 
                                fast = False, 
                                limits1 = lim1,
                                limits2 = lim2,  
                                L1R = 1000, 
                                L2R = None,
                                Param = "shift",
                                procpars = {'SSB':2}
                                )
    
    for J in range(150,401,25):
        A_h = np.ones(8)*0.8

        dia_shift = np.array([40.5, 58, 40.5, 58, 58, 40.5, 58, 58])

        dia_shift = np.tile(dia_shift, (len(temp),1))
        shift_tot = shift_tot-dia_shift
        print(shift_tot)
        param = lmfit.Parameters()
        param.add('J', value=J, min=0, max=1000, vary=False)
        [param.add('A_'+str(i), value=A_h[i], min=0, max=2, vary=True) for i in range(len(A_h))]
        print('shift tot\n',shift_tot)
        minner = lmfit.Minimizer(T_model, param, fcn_args=(temp, shift_tot))
        result = minner.minimize(method='leastsq', max_nfev=30000)
        popt = result.params
        print(popt.pretty_print())  
        print(lmfit.fit_report(result))
        calc_shift = T_model(popt, temp, shift_tot, result=True, T_long=np.arange(200, 20000, 10))
        print(calc_shift)