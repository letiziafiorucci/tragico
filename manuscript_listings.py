

# THIS FILE CONTAINS ALL THE EXAMPLE SCRIPT PRESENT IN THE PAPER 
# it is not executable
# just copy the code you need in your script

###### LISTING 1 ######

from f_fit import *
import numpy as np


path = 'path/to/spectra/folder/'


# in this case the experiments’ names correspond to consecutive numbers (from 2 to 14)
num_sp = list(np.arange(2,15,1))  
list_sp = [str(i)+'/pdata/1' for i in num_sp]


temp = []
for idx in range(len(list_sp)):
   _, _, ngdicp = nmr_spectra_1d(path+list_sp[idx])
   temp.append(ngdicp['acqus']['TE1'])
temp = np.array(temp)


# shift (ppm), lw (ppm), x_g (adim.), k (adim.), ph (rad), A-B-C-D-E (a.u.)
lim1 = {'shift':(-1,1), 'lw':(1e-4,2.5), 'ph':(-np.pi/20,np.pi/20),'A':(0,0), 'B':(0,0), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
lim2 = {'shift':(0.9,1.1), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)}


_, _, (shift_tot, shift_tot_err), temp, _ = model_fit_1D(
                                                        path,
                                                        temp,
                                                        list_sp,
                                                        dofit=True,
                                                        fast = True,
                                                        limits1 = lim1,
                                                        limits2 = lim2,
                                                        file_inp1=None, 
                                                        file_inp2=None,
                                                        prev_fit=None,
                                                        L1R = 0.0,
                                                        L2R = 0.0, 
                                                        Param='shift')


###### LISTING 2 ######

from f_fit import *
import numpy as np


path = 'path/to/spectra/folder/'


# the example covers one of our time series containing 368 experiments, starting from 101 and ending with 468


num_sp = list(np.arange(101, 469, 1)) 
list_sp = [str(i)+'/pdata/1' for i in num_sp]


delays = np.linspace(0, 1, len(list_sp))


dir_result = intensity_fit_1D(
                            path,
                            delays,
                            list_sp,
                            area=True,
                            cal_lim = (-68.00,-78.00),
                            baseline=True, # if True, the baseline is subtracted
                            delta = 5
                            )


###### LISTING 3 ######

from f_fit import *
import numpy as np

# folder containing the spectra
path = 'path/to/TRAGICO_analysis/olive_oil_experiment/'


intensity_fit_pseudo2D(
                       path,
                       delays_list=[np.loadtxt(path+'1/vdlist')],
                       list_path=['1'],                                                                                      
                       area=True,   # True- Integration, False - Intensity
                       VCLIST=[1.0],
                       delta = 0.2,
                       cal_lim = (2.5,2.3), 
                       baseline=True, 
                       doexp=False,
                       f_int_fit = fit_IR,
                       fargs={}, 
                       fileinp=None,
                       IR=True, # Inverts the order of the delays to make the interactive guess on the longest delay 
                       err_lims_out= None, # (ppm1, ppm2)
                       color_map = 'hsv')


###### LISTING 4 ######

from f_fit import *
import numpy as np


path = 'path/to/TRAGICO_analysis/olive_oil_experiment/'


list_path = [int(f) for f in os.listdir(path) if not f.startswith('.')]
list_path = [str(f) for f in np.sort(list_path)]


delays_list = [np.loadtxt(path+list_path[i]+'/vdlist')+3e-3 for i in range(len(list_path))]


lim1 = {'shift':(-0.05,0.05), 'lw':(1e-4,0.05), 'ph':(-np.pi/20,np.pi/20), 'k':(-1,1), 'B':(0,0), 'C':(0,0), 'D':(0,0), 'E':(0,0)}
lim2 = {'shift':(0.95,1.05), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)}


model_fit_pseudo2D(
                path, 
                delays_list,
                list_path, 
                cal_lim = (2.5,2.3), # limits for the calibration
                file_inp1 = None,
                file_inp2 = None,
                fast = True, 
                VCLIST=[1.0],
                dofit = True,
                limits1 = lim1, 
                limits2 = lim2,
                L1R = 0.0,
                L2R = 0.0,
                f_int_fit = fit_IR,
                fargs={}, 
                IR=True, # Inverts the order of the delays to make the interactive guess on the longest delay
                doexp=False)


###### LISTING S1.2.2 ######

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


   normalization = np.max(data[0,:])


   def residue(param, ppm_scale, spettro0, spettro1, sx, dx, risultato=False):
       par = param.valuesdict()
       roll_spettro1 = np.roll(spettro1, int(par['shift']))
       if not risultato:
           res = np.abs(spettro0.real/normalization)-np.abs(roll_spettro1.real/normalization)
           return res[npoints:-npoints]
       else:
           if debug_fig:
               fig = plt.figure()
               fig.set_size_inches(5.59,4.56)
               plt.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.90)
               ax = fig.add_subplot(1,1,1)
               ax.tick_params(labelsize=6.5)
               ax.plot(ppm_scale[sx:dx], spettro0.real/normalization, lw=0.5, label='spectra_0')
               ax.plot(ppm_scale[sx:dx], roll_spettro1.real/normalization, lw=0.5, label='spectra_1')
               ax.plot(ppm_scale[sx:dx], spettro1.real/normalization, lw=0.5, label='spectra_1 prima')
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
           result = minner.minimize(method='brute', max_nfev=1000)           popt = result.params
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


###### LISTING S2.3.1 ######

import matplotlib.pyplot as plt


delta_ppm = shift_tot[:,1]-shift_tot[:,0]
delta_ppm_err = np.sqrt(shift_tot_err[:,0]**2+shift_tot_err[:,1]**2)
temp_err = 0.025  # in K
temp_corr = 416.4745 - (38.5133)*delta_ppm - 36.0620*((delta_ppm)**2) + 11.4869*((delta_ppm)**3) - 2.4340*((delta_ppm)**4)
fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(4.,3.5)
ax.errorbar(temp_corr, delta_ppm, xerr=temp_err, yerr=delta_ppm_err, fmt='.', color='darkred', ecolor='m', elinewidth=0.2, capsize=2, capthick=0.2)
ax.set_xlabel('T$_{corr}$ (K)')
ax.set_ylabel('$\Delta \delta$ (ppm)')
plt.tight_layout()
plt.savefig('temp_corr.png', dpi=600)
plt.show()


np.savetxt('/home/methanol_modelfit/temp_corr.txt', temp_corr)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(4.,3.5)
#trend line
y = np.polyfit(temp, temp_corr, 1)
y = np.poly1d(y)
#write the equation of the line
ax.plot(temp, y(temp), '--', color='blue', label=f'y = {y[1]:.4f}x + {y[0]:.4f}')
ax.plot(temp, temp_corr, 'o', color='magenta')
ax.set_xlabel('T$_{set}$ (K)')
ax.set_ylabel('T$_{corr}$ (K)')
plt.tight_layout()
plt.savefig('temp_corr_trend.png', dpi=600)
plt.legend()
plt.show()


###### LISTING S2.4.1 ######

from f_fit import *
import numpy as np


path = 'path/to/spectra/folder/Nisal'


num_sp = list(np.arange(10,22,1)) # in this case the experiments are enumerated from 10 to 21 
list_sp = [str(i)+'/pdata/1' for i in num_sp]


temp = []
for idx in range(len(list_sp)):
   _, _, ngdicp = nmr_spectra_1d(path+list_sp[idx])
   temp.append(ngdicp['acqus']['TE1'])
temp = np.array(temp)


# shift (ppm), lw (ppm), x_g (adim.), k (adim.), ph (rad), A-B-C-D-E (a.u.)
lim1 = {'shift':(-1,1), 'lw':(1e-4,12), 'ph':(-np.pi/20,np.pi/20), 'D':(0,0), 'E':(0,0)}
lim2 = {'shift':(0.2,1.8), 'lw':(0,0), 'ph':(0,0), 'xg':(0,0)}


_, shift_tot, x = model_fit_1D(
           path, # path of the main spectra
           temp, # list of temperatures
           list_sp, # list of folders
           dofit=True, # if True, the spectra are fitted
           fast = True, 
           limits1 = lim1,  
           limits2 = lim2,   
           Param='shift')


###### LISTING S2.7.1 ######

from scipy.stats import chi2
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


data_path = "/home/Nisal_modelfit/Param.txt"
temp_es_path = "/home/Nisal_modelfit/x_1.txt"


data = np.genfromtxt(data_path)
temp_es = np.genfromtxt(temp_es_path)


errors_path = "/home/Nisal_modelfit/Param_err.txt"
errors = np.genfromtxt(errors_path)


peaks = {f"pk{i+1}": data[:, i] for i in range(19)}


temp_set_MeOH_path ='/home/methanol_modelfit/x_1.txt'
temp_set_MeOH = np.genfromtxt(temp_set_MeOH_path)


# correct temperature MeOH
temp_corr_MeOH_path = '/home/methanol_modelfit/temp_corr.txt'
temp_corr_MeOH = np.genfromtxt(temp_corr_MeOH_path)


x1 = temp_set_MeOH
y1 = temp_corr_MeOH


coefficients = np.polyfit(x1, y1, 1) 
a, b = coefficients
temp_corr_nisal = a*temp_es + b


inv = 1/temp_corr_nisal
temp_err = 0.025
new_xerr = temp_err / (temp_corr_nisal**2)


ppp = [peaks[f"pk{i}"] for i in range(1, 20)]  # Extract pk1, pk2, ..., pk19


for i, y in enumerate(ppp): 
   fig = plt.figure()  # Create a new figure for each array
   ax = fig.add_subplot(1, 1, 1)
   fig.set_size_inches(4.5, 4)
  
   # Plot data with error bars
   plt.errorbar(inv, y, xerr=new_xerr, yerr=errors[:, i], fmt='.', color='darkgreen',
                ecolor='m', elinewidth=0.9, capsize=2, capthick=0.2)
   plt.title(f"Peak {i+1}")
  
   # Fit data to y = ax + b
   aa, bb = np.polyfit(inv, y, 1)  # 1 for linear fit  
  
   # Compute the fitted values
   yfit = aa * inv + bb


    # Calculate R^2 score
   r2 = r2_score(y, yfit)


   plt.errorbar(inv, yfit, color='orange', label=f"y = {aa:.2e}x + {bb:.2e}\n$R^2$ = {r2:.4f}")
     
   ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
   ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
   plt.xlabel("$1/T_{corr}$ (1/K)", fontsize=12)
   plt.ylabel("$\delta$ (ppm)", fontsize=12)
   plt.legend()
   plt.tight_layout()
  
   fig.savefig(f"plot_{i+1}_nisal.png", dpi=600, bbox_inches='tight')
   plt.show() 


###### LISTING S3.1.1 ######

from f_fit import *
import numpy as np


def cal_temp(T):
    #temperature calibration at 1200 MHz
    T_cal = []
    for t in T:
        T_cal.append(t*0.955944+9.81982517+2)
    return T_cal


path = 'path/to/spectra/'
num_sp = list(np.arange(23,31,1))   
list_sp = [str(i)+'/pdata/10' for i in num_sp]


temp = []
for idx in range(len(list_sp)):
   _, _, ngdicp = nmr_spectra_1d(path+list_sp[idx])
   temp.append(ngdicp['acqus']['TE'])


temp = cal_temp(temp)  #calibration of the temperature at 1200 MHz
temp = np.array(temp)


lim1 = {'shift':(-4,4), 'lw':(1e-4,2.5), 'ph':(-np.pi/10,np.pi/10), 'k':(0,1)}
lim2 = {'shift':(0.5,1.5),'k':(0,0),'lw':(0,0), 'ph':(0,0), 'xg':(0,0), 'A':(0.9,1.1), 'B':(0.9,1.1), 'C':(0.9,1.1), 'D':(0.9,1.1), 'E':(0.9,1.1)}


dir_res, _, shift_tot, temp, *_ = model_fit_1D(
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
shift_tot = shift_tot[0]


###### LISTING S3.2.1 ######

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import lmfit


def calc_shift(temp, S2, S3, A_h, J, gammaC=267/4):


   def energy(Si, J):
       return 1/2*J*(Si*(Si+1))


   conv = 1.9865e-23 # from cm-1 to J
   kB = scipy.constants.k
   muB = scipy.constants.physical_constants['Bohr magneton'][0]
   ge = 2.0023
   pref = 2*np.pi*ge*muB/(3*kB*gammaC*temp)


   sum1 = 0
   sum2 = 0
   for s in np.arange(np.abs(S2-S3), S2+S3+1, 1):
       sum1 += 1/2*s*(s+1)*(2*s+1)*np.exp(-energy(s, J*conv)/(kB*temp))
       sum2 += (2*s+1)*np.exp(-energy(s, J*conv)/(kB*temp))


   sum1 *= pref*A_h*1e6
   shift = sum1/sum2


   return shift


def T_model(param, temp, shift, result=False, T_long=None, name=None):
  
   S = 5/2
   par = param.valuesdict()
   J = par['J']
   A = [par['A_'+str(i)] for i in range(shift.shape[1])]
   res = []
   conshift = np.zeros((len(temp), shift.shape[1]))
   for i in range(len(temp)):
       for j in range(shift.shape[1]): 
           conshift_s = calc_shift(temp[i], S, S, A[j], J)
           conshift[i,j] = conshift_s
           res.append(conshift_s-shift[i,j])


   if result:
       cont_list = []
       for j in range(shift.shape[1]):
           cont_list.append([calc_shift(t, S, S, A[j], J) for t in T_long])
       fig = plt.figure()
       ax = fig.add_subplot(111)
       fig.set_size_inches(8, 10)
       for i in range(shift.shape[1]):
           ax.plot(1000/T_long, cont_list[i], lw = 0.7, c='k')
       for i in range(shift.shape[1]):
           line, = ax.plot(1000/temp, shift[:,i], 'o', label=f'{i+1}: J = {J:.1f} '+r'cm$^{-1}$'+f'; A/h = {A[i]:.3f} MHz')
           ax.plot(1000/temp, conshift[:,i], '--', c=line.get_color())


       plt.legend()
       plt.xlabel(r'1000/T (K$^{-1}$)', fontsize=13)
       plt.ylabel('Contact shift (ppm)', fontsize=13)
       ax.tick_params(axis='both', which='major', labelsize=13)
       plt.savefig(name+'.png', dpi=600)
       plt.close()
       return conshift
   else:
       return res


J = 300 # cm^-1
A_h = np.ones(8)*0.8
dia_shift = np.array([26.8, 56.27, 29.13, 24.4, 59.8, 27.8, 58, 57.24])


param = lmfit.Parameters()
param.add('J', value=J, min=0, max=1000, vary=False)
[param.add('A_'+str(i), value=A_h[i], min=0, max=3, vary=True) for i in range(shift_tot.shape[1])]
minner = lmfit.Minimizer(T_model, param, fcn_args=(temp, shift_tot))
result = minner.minimize(method='leastsq', max_nfev=30000)
popt = result.params
T_model(popt, temp, shift_tot, result=True, T_long=np.arange(200, 20000, 10), name=dir_res+'/'+'J_'+str(J))


###### LISTING S4.4 ######

import matplotlib.pyplot as plt
import numpy as np


datax = np.arange(101, 397, 1).tolist() #the range of experiments that are being plotted, in our case we start from experiment 101 and end at 397
datay = np.loadtxt("<folder>_modelfit/y_1.txt")[::-1]
dataz = np.loadtxt("<folder>_modelfit/y_2.txt")[::-1]


plt.plot(datax, datay, "g.") #plotting the first trend with green-coloured points
plt.plot(datax, dataz, "m.") #plotting the second trend with magenta-coloured points
plt.xlabel("Experiment number", fontsize = "14")
plt.ylabel("Integral (a.u.)", fontsize = "14")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText = True)
plt.savefig("your_modelfit_integrals.png", transparent=False, dpi=600, format=None, metadata=None, bbox_inches=None, pad_inches=0.1, facecolor="auto", edgecolor="auto", backend=None) #saving the figure in .png format
plt.show()
exit()


###### LISTING S4.5 ######

sumcomp = None

if result:
    onecomp = lor.copy()
    onecomp *= em(onecomp, LB, SW)
    onecomp *= qsin(onecomp, SSB)
    onecomp = zf(onecomp, 16384)
    onecomp = ft(onecomp, SI, dw, o1p, sf1)[0]
    sumcomp = np.zeros_like(onecomp)
    sumcomp += onecomp
    np.save("your_16k_point_hard_model.npy", sumcomp)


###### LISTING S5.3 ######

def fit_IR(x, y, name= None, err_bar=None, figura=True):

   def IR_curve(t, t1):
       return 1-2*np.exp(-t/t1)

   def IR_residue(param, t, exp, result=False):
       par = param.valuesdict()
       t1 = 10**par['t1']
       model = IR_curve(t, t1)
       den = np.mean(model**2)-np.mean(model)**2
       a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
       A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
       if not result:
           return (a + A*model)-exp
       else:
           return a+A*model, A, a

   param = lmfit.Parameters()
   param.add('t1', value=0, min=-4, max=1)
   minner = lmfit.Minimizer(IR_residue, param, fcn_args=(x, y))
   result1 = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-8, ftol=1e-8)
   popt1 = result1.params
   report1 = lmfit.fit_report(result1)

   func1, A, a = IR_residue(popt1, x, y, result=True)
   popt1.add('A', value=A)
   popt1.add('a', value=a)

   #compute the mean squared deviation of the experimental points from the model function
   RMSE1 = np.sqrt(np.mean((y-func1)**2))

   fig=plt.figure()
   fig.set_size_inches(3.59,2.56)
   ax = fig.add_subplot(1,1,1)
   ax.tick_params(labelsize=12)
   ax.plot(x, y, 'o', c='k', markersize=4)
   #ax.set_xscale('log')
   if err_bar is not None:
       ax.errorbar(x, y, yerr=err_bar, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
   label_mod = ''
   for key, values in popt1.items():
       if key=='t1':
           try:
               label_mod+=f'{10**values.value:.5e}'+r' $\pm$ '+f'{np.abs(10**values.value*values.stderr/values.value):.5e}'
               err1 = np.abs(10**values.value*values.stderr/values.value)
           except:
               label_mod+=f'{10**values.value:.5e}'+r' $\pm$ Nan'
               err1 =  None

   ax.plot(x, func1, 'r--', lw=1, label='y = a + A (1 - 2 exp(-t/T1))\nfit: T1 = '+label_mod+' sec\n'+r'$\chi^2$ red: '+f'{result1.redchi:.5e}'+r', RMSE: '+f'{RMSE1:.5e}')
   ax.set_xlabel('Delay (s)', fontsize=12)
   ax.set_ylabel('Intensity (a.u.)', fontsize=12)
   ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3,3), useMathText=True)
   ax.yaxis.get_offset_text().set_size(11)
   ax.legend(fontsize=10)
   plt.tight_layout()
   if figura:
       if name is None:
           plt.show()
       else:
           plt.savefig(name+'.png', dpi=600)
           plt.close()

       return popt1.valuesdict(), func1, report1, err1, RMSE1


