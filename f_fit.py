
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from f_functions import *
from pprint import pprint
import os
import shutil


def intensity_fit_pseudo2D(path, delays_list, list_path, prev_lims = False, prev_coeff = False, area=False, auto_ph=False, VCLIST=None, cal_lim = None, baseline=False, delta=0, doexp=False, f_int_fit=None, fargs=None):

    # series of pseudo2D spectra 

    # path: path to the directory containing the spectra
    # delays_list: list of parameters to be used as x-variable in the fit of the intensity (can be time, temperature, etc.)
    # list_path: list of pseudo2Ds
    # area: if True the integral of the peak is calculated, otherwise the intensity is the maximum value of the peak
    # VCLIST: list of the parameters (for HRR are the field values) that differentiate one pseudo2D from the other
    # cal_lim: limits of the calibration region in ppm given as an iterable (e.g. [0.0, 10.0])
    # baseline: if True the baseline is subtracted
    # f_int_fit: function for the intensity fit against the chosen x-variable (delays_list)
    # doexp: if True the intensities are fitted with an exponential decay (in this case f_int_fit is not used)
    # fargs: dictionary of arguments to be passed to f_int_fit

    for i in range(len(delays_list)):
        if 'pdata' not in list_path[i]:
            list_path[i] = list_path[i]+'/pdata/1'

    new_dir = os.path.basename(os.path.normpath(path))
    if area:
        nome_folder = '_integral'
    else:
        nome_folder = '_intensity'
    
    try:
        os.mkdir(new_dir+nome_folder)
    except:
        ans = input('\nThe directory '+color_term.BOLD+new_dir+nome_folder+color_term.END+' already exists.\nDo you want to:\n[overwrite the existing directory (1)]\ncreate a new directory (2)\ncontinue on writing on the existing (3)\n>')
        if ans=='1' or ans == '':
            shutil.rmtree(new_dir+nome_folder)           # Removes all the subdirectories!
            os.makedirs(new_dir+nome_folder)
        elif ans=='2':
            new_dir = input('Write new directory name: ')
            os.makedirs(new_dir+nome_folder)
        elif ans=='3':
            pass
        else:
            print('\nINVALID INPUT')
            exit()
    print('Writing directory '+color_term.BOLD+new_dir+nome_folder+color_term.END)

    #creats directorys inside dir_result with the same names as the spectra in path
    dir_result = new_dir+nome_folder
    [os.makedirs(dir_result+'/'+list_path[i][:list_path[i].index('/pdata')]) for i in range(len(list_path))]
    dir_result_sp = [dir_result+'/'+list_path[i][:list_path[i].index('/pdata')] for i in range(len(list_path))]  # 'nome_directory_risultato/numero_spettro'

    if VCLIST is None:
        VCLIST = []
    field = False
    for idx, dir_res in enumerate(dir_result_sp):  #for every pseudo2D

        print('DIR: ', color_term.CYAN+dir_res+color_term.END)

        nameout = new_dir+'_sp'+list_path[idx][:list_path[idx].index('/pdata')]+'.out'

        delays = delays_list[idx]

        print('PATH TO SPECTRA: ', path+'/'+list_path[idx])

        datap, ppm_scale, dic = nmr_spectra_pseudo2d(path+'/'+list_path[idx]) #(n.delay x TD)

        title = open(path+'/'+list_path[idx]+'/title').readlines()  #read field from title: format xxx.xxmT (this works only for HRR measurements)
        if VCLIST==[] or field==True:
            for i,line in enumerate(title):
                if 'The selected field' in line:
                    line = line.replace('\n','')
                    splitline = line.split(' ')
                    for ii in range(len(splitline)):
                        try:
                            field = float(splitline[ii][:-2])
                        except:
                            pass

            VCLIST.append(field)
            field = True

        data = np.array([datap[r,:] for r in range(datap.shape[0]) if sum(datap[r,:]) != 0+1j*0])  #tolgo eventuali spettri non acquisiti
        print('DATA SHAPE: ', data.shape)

        if auto_ph:
            data_p = []
            for i in range(data.shape[0]):
                p0, p1 = acme(data[i,:])
                data = ps(data[i,:], ppm_scale, p0, p1)
                data_p.append(data[i,:].real)
            data = np.array(data_p)

        to_order = np.hstack((np.reshape(delays,(len(delays),1)),data))
        to_order = to_order[to_order[:,0].argsort()]#[::-1]]

        data = to_order[:,1:]
        delays = to_order[:,0].real

        # performs calibration on the spectra if cal_lim is not None
        if cal_lim is not None:
            cal_shift, cal_shift_ppm, data = calibration(ppm_scale, data, cal_lim[0], cal_lim[1]) 
            with open(dir_res+'/'+nameout, 'w') as f:
                f.write('SPECTRA PATH: \n')
                f.write(path+'/'+list_path[idx]+'\n')
                f.write('\n')
                f.write('CALIBRATION: ('+f'{cal_lim[0]:.5f}'+':'+f'{cal_lim[1]:.5f}'+') ppm\n')
                f.write('in points\n')
                [f.write(str(cal_shift[ii])+'   ') for ii in range(len(cal_shift))]
                f.write('\nin ppm\n')
                [f.write(f'{cal_shift_ppm[ii]:.5f}'+'   ') for ii in range(len(cal_shift_ppm))]
                f.write('\n\n')
                f.write('Points:\n')
                [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
                if VCLIST is not None and len(VCLIST)>0:
                    f.write('\n')
                    f.write('VCLIST point: '+f'{VCLIST[idx]:.2f}'+' T\n')
                f.close()
        else:
            with open(dir_res+'/'+nameout, 'w') as f:
                f.write('SPECTRA PATH: \n')
                f.write(path+'/'+list_path[idx]+'\n')
                f.write('\n')
                f.write('Points:\n')
                [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
                if VCLIST is not None and len(VCLIST)>0:
                    f.write('\n')
                    f.write('VCLIST point: '+f'{VCLIST[idx]:.2f}'+' T\n')
                f.close()

        spettro = data[0,:]

        CI = create_input(ppm_scale, spettro)
        if idx>0 and not prev_lims:
            ans = input('Do you want to select different regions? ([y]|n) ')
            if ans=='y' or ans=='':
                limits, err_lims = CI.select_regions()
            else:
                pass
        elif idx==0:
            limits, err_lims = CI.select_regions()
        elif idx>0 and prev_lims:
            pass

        int_tot = []
        coeff_list = []
        for k in range(len(limits)):
            
            if baseline:
                if idx>0 and not prev_coeff:
                    ans = input('Do you want to select different baseline coefficients for region ('+f'{limits[k][0]:.3f}, {limits[k][1]:.3f}'+') ppm? ([y]|n) ')
                    if ans=='y' or ans=='':
                        coeff = CI.make_iguess(limits=[limits[k][0]-delta, limits[k][1]+delta], true_limits=[limits[k][0], limits[k][1]])   
                    else:
                        coeff = old_coeff[k]
                elif idx==0:
                    coeff = CI.make_iguess(limits=[limits[k][0]-delta, limits[k][1]+delta], true_limits=[limits[k][0], limits[k][1]])
                elif idx>0 and prev_coeff:
                    coeff = old_coeff[k]
            else:
                coeff = np.zeros(5)
            coeff_list.append(coeff)

            sx, dx, zero = find_limits(limits[k][0], limits[k][1], ppm_scale)  

            x = ppm_scale[sx:dx]-zero
            A,B,C,D,E = coeff
            corr_baseline = E*x**4 + D*x**3 + C*x**2 + B*x + A

            intensity = []
            for iii in range(len(delays)):
                if area:
                    intensity.append(np.trapz(data[iii,sx:dx].real - corr_baseline))
                else:
                    intensity.append(np.max(data[iii,sx:dx].real - corr_baseline))

            int_tot.append(intensity)

        old_coeff = coeff_list.copy()
                    
        
        with open(dir_res+'/'+nameout, 'a') as f:
            f.write('\n')
            f.write('Selected intervals (ppm):\n')
            for ii in range(len(limits)):
                f.write(str(ii+1)+'\t'+f'{limits[ii][0]:.4f}'+'\t'+f'{limits[ii][1]:.4f}'+'\n')
            f.close()

        integral=np.array(int_tot).T
        Coeff = np.array(coeff_list)

        #error evaluation
        if area:
            sx, dx, zero = find_limits(err_lims[0], err_lims[1], ppm_scale)
            error = []
            for k in range(len(limits)):
                error.append([])
                sxi, dxi, _ = find_limits(limits[k][0], limits[k][1], ppm_scale) 
                for iii in range(len(delays)):
                    error[k].append(np.mean(np.abs(data[iii,sx:dx].real))*(dxi-sxi)) 
            error = np.array(error).T
        else:
            sx, dx, zero = find_limits(err_lims[0], err_lims[1], ppm_scale)
            error = []
            for iii in range(len(delays)):
                error.append(np.std(data[iii,sx:dx].real))

        int_del = np.column_stack((integral, delays))  #(n. delays x [integral[:,0],...,integral[:,n], delays[:]])
        order = int_del[:,-1].argsort()
        int_del = int_del[order]
        if area:
            error = np.array(error)[order,:]
        else:
            error = np.array(error)[order]

        with open(dir_res+'/'+nameout, 'a') as f:
            f.write('\n')
            for r in range(100):
                f.write('=')
            f.write('\n')
            f.write('\n')
            if not area:
                np.savetxt(dir_res+'/Err.txt', error)
            for j in range(integral.shape[1]):
                if area:
                    np.savetxt(dir_res+'/Err_'+str(j+1)+'.txt', error[:,j])
                np.savetxt(dir_res+'/y_'+str(j+1)+'.txt', integral[:,j])
                np.savetxt(dir_res+'/x_'+str(j+1)+'.txt', delays)
                f.write('N. interval: '+str(j+1)+'\n')
                if baseline:
                    f.write('Baseline coefficients\n')
                    f.write('A\t'+f'{Coeff[j,0]:.5e}'+'\n')
                    f.write('B\t'+f'{Coeff[j,1]:.5e}'+'\n')
                    f.write('C\t'+f'{Coeff[j,2]:.5e}'+'\n')
                    f.write('D\t'+f'{Coeff[j,3]:.5e}'+'\n')
                    f.write('E\t'+f'{Coeff[j,4]:.5e}'+'\n')
                if area:
                    f.write('N. point\tIntegral\n')
                else:
                    f.write('N. point\tIntensity\n')
                for i in range(len(delays)):
                    if area:
                        f.write(str(i)+'\t'+f'{integral[i,j]:.3f}'+' +/- '+f'{error[i,j]:.3f}'+'\n')
                    else:
                        f.write(str(i)+'\t'+f'{integral[i,j]:.3f}'+' +/- '+f'{error[i]:.3f}'+'\n')
                f.write('\n')

        if doexp==True:

            n_peak = 0
            mono_fit = []
            bi_fit = []
            errmono_fit = []
            errbi_fit = []
            fitparameter_f = []
            for ii in range(int_del.shape[-1]-1):
                n_peak += 1
                if area:
                    mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}',err_bar=error[:,ii])
                else:
                    mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}',err_bar=error)
            
                
                with open(dir_res+'/'+nameout, 'a') as f:
                    f.write('\n')
                    [f.write('-') for r in range(30)]
                    f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                    [f.write('-') for r in range(30)]
                    f.write('\n')
                    f.write('\nFit Parameters:\n')
                    f.write('\nMono\n')
                    f.write('y = a + A exp(-t/T1)\nfit: T1=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(mono))
                    f.write(report1+'\n')
                    f.write('\nBi\n')
                    f.write('y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: f=%5.3e, T1a=%5.4e, T1b=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(bi))
                    f.write(report2+'\n')
                    f.close()

                mono_fit.append(10**tuple(mono)[0])
                errmono_fit.append(err1)
                bi_fit.append((10**tuple(bi)[1], 10**tuple(bi)[2]))
                errbi_fit.append(err2)
                fitparameter_f.append(tuple(bi)[0])
                
                
            with open(dir_res+'/'+'t1.txt', 'w') as f:
                f.write('n.peak\tT1 (s)\terr (s)\tf\n')
                for ii in range(len(mono_fit)):
                    try:
                        f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+f'{errmono_fit[ii]:.4e}'+'\n')
                    except:
                        f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+'Nan'+'\n')
                    try:
                        f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+f'{errbi_fit[ii][0]:.4e}'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+f'{errbi_fit[ii][1]:.4e}'+'\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                    except:
                        f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+'Nan'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+'Nan\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                f.close()
            with open(dir_res+'/'+nameout, 'a') as f:
                for r in range(100):
                    f.write('=')
                f.close()

        elif f_int_fit is not None and doexp==False:

            n_peak = 0
            for ii in range(int_del.shape[-1]-1):
                n_peak += 1
                #f_int_fit must return a dictionary with the fit parameters and the model
                parameters, model = f_int_fit(int_del[:,-1], int_del[:,ii], **fargs, err_bar = error)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
                ax.plot(int_del[:,-1], model, 'r--', lw=0.7)
                if area:
                    ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error[:,ii], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                    ax.set_ylabel('Integral')
                else:
                    ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                    ax.set_ylabel('Intensity')
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
                plt.savefig(dir_res+'/Interval_'+str(ii+1)+'_sp'+str(idx+1)+'.png', dpi=600)
                plt.close()

                with open(dir_res+'/'+nameout, 'a') as f:
                    f.write('\n')
                    [f.write('-') for r in range(30)]
                    f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                    [f.write('-') for r in range(30)]
                    f.write('\n')
                    f.write('\nFit Parameters:\n')
                    f.write('\n')
                    f.write(' '.join([f'{key}={val}' for key, val in parameters.items()])+'\n')
                    f.close()


        else:

            for ii in range(int_del.shape[-1]-1):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
                if area:
                    ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error[:,ii], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                    ax.set_ylabel('Integral')
                else:
                    ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                    ax.set_ylabel('Intensity')
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
                plt.savefig(dir_res+'/Interval_'+str(ii+1)+'_sp'+str(idx+1)+'.png', dpi=600)
                plt.close()

        

    if len(VCLIST)>0:        
        VCLIST = np.array(VCLIST)
        np.savetxt(dir_result+'/VCLIST.txt', VCLIST)
            
    return dir_result

def intensity_fit_1D(path, delays_list, list_path, area=False, auto_ph=False, cal_lim = None, baseline=False, delta=0, doexp=False, f_int_fit=None, fargs=None):

    # series of 1D spectra to be treated as a pseudo2D

    # path: path to the directory containing the spectra
    # delays_list: list of parameters to be used as x-variable in the fit of the intensity (can be time, temperature, etc.)
    # list_path: list of the names of the spectra to be stacked in a pseudo2D
    # area: if True the integral of the peak is calculated, otherwise the intensity is the maximum value of the peak
    # cal_lim: limits of the calibration region in ppm given as an iterable (e.g. [0.0, 10.0])
    # baseline: if True the baseline is subtracted
    # f_int_fit: function for the intensity fit against the chosen x-variable (delays_list)
    # doexp: if True the intensities are fitted with an exponential decay (in this case f_int_fit is not used)
    # fargs: dictionary of arguments to be passed to f_int_fit

    for i in range(len(delays_list)):
        if 'pdata' not in list_path[i]:
            list_path[i] = list_path[i]+'/pdata/1'

    new_dir = os.path.basename(os.path.normpath(path))
    if area:
        nome_folder = '_integral'
    else:
        nome_folder = '_intensity'
    
    try:
        os.mkdir(new_dir+nome_folder)
    except:
        ans = input('\nThe directory '+color_term.BOLD+new_dir+nome_folder+color_term.END+' already exists.\nDo you want to:\n[overwrite the existing directory (1)]\ncreate a new directory (2)\ncontinue on writing on the existing (3)\n>')
        if ans=='1' or ans == '':
            shutil.rmtree(new_dir+nome_folder)           # Removes all the subdirectories!
            os.makedirs(new_dir+nome_folder)
        elif ans=='2':
            new_dir = input('Write new directory name: ')
            os.makedirs(new_dir+nome_folder)
        elif ans=='3':
            pass
        else:
            print('\nINVALID INPUT')
            exit()
    print('Writing directory '+color_term.BOLD+new_dir+nome_folder+color_term.END)

    dir_res = new_dir+nome_folder
    nameout = new_dir+'_sp'+dir_res+'.out'
    
    # data_d = []
    # ppmscale_d = []
    # for i in range(len(delays_list)):
    #     data, ppmscale, _ = nmr_spectra_1d(path+list_path[i])
    #     plt.plot(ppmscale, data)
    #     if auto_ph:
            
    #         p0, p1 = acme(data)
    #         data = ps(data, ppmscale, p0, p1)[0]
    #     plt.plot(ppmscale, data)
    #     plt.show()
    #     data_d.append(data)
    #     ppmscale_d.append(ppmscale)

    data_d = []
    ppmscale_d = []
    for i in range(len(delays_list)):
        data, ppmscale, _ = nmr_spectra_1d(path+list_path[i])
        data_d.append(data)
        ppmscale_d.append(ppmscale)

    data = np.array(data_d)
    ppm_scale = ppmscale.copy()  

    if cal_lim is not None:
        cal_shift, cal_shift_ppm, data = calibration(ppm_scale, data, cal_lim[0], cal_lim[1]) 
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('CALIBRATION: ('+f'{cal_lim[0]:.5f}'+':'+f'{cal_lim[1]:.5f}'+') ppm\n')
            f.write('in points\n')
            [f.write(str(cal_shift[ii])+'   ') for ii in range(len(cal_shift))]
            f.write('\nin ppm\n')
            [f.write(f'{cal_shift_ppm[ii]:.5f}'+'   ') for ii in range(len(cal_shift_ppm))]
            f.write('\n\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays_list[r]:.3f}'+'\n') for r in range(len(delays_list))]
            f.close()
    else:
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays_list[r]:.3f}'+'\n') for r in range(len(delays_list))]
            f.close()

    spettro = data[0,:] #change this if you what to make the guess from another spectrum

    CI = create_input(ppm_scale, spettro)
    limits, err_lims = CI.select_regions()

    int_tot = []
    coeff_list = []
    for k in range(len(limits)):

        if baseline:
            coeff = CI.make_iguess(limits=[limits[k][0]-delta, limits[k][1]+delta], true_limits=[limits[k][0], limits[k][1]])
        else:
            coeff = np.zeros(5)
        coeff_list.append(coeff)

        sx, dx, zero = find_limits(limits[k][0], limits[k][1], ppm_scale)  

        x = ppm_scale[sx:dx]-zero
        A,B,C,D,E = coeff
        corr_baseline = E*x**4 + D*x**3 + C*x**2 + B*x + A

        intensity = []
        for iii in range(len(delays_list)):
            if area:
                intensity.append(np.trapz(data[iii,sx:dx].real - corr_baseline))
            else:
                intensity.append(np.max(data[iii,sx:dx].real - corr_baseline))

        int_tot.append(intensity)

    Coeff = np.array(coeff_list)

    #error evaluation
    if area:
        sx, dx, zero = find_limits(err_lims[0], err_lims[1], ppm_scale)
        error = []
        for k in range(len(limits)):
            error.append([])
            sxi, dxi, _ = find_limits(limits[k][0], limits[k][1], ppm_scale) 
            for iii in range(len(delays_list)):
                error[k].append(np.mean(np.abs(data[iii,sx:dx].real))*(dxi-sxi)) 
        error = np.array(error).T
    else:
        #error evaluation
        sx, dx, zero = find_limits(err_lims[0], err_lims[1], ppm_scale)
        error = []
        for iii in range(len(delays_list)):
            error.append(np.std(data[iii,sx:dx].real))
    
    with open(dir_res+'/'+nameout, 'a') as f:
        f.write('\n')
        f.write('Selected intervals (ppm):\n')
        for ii in range(len(limits)):
            f.write(str(ii+1)+'\t'+f'{limits[ii][0]:.4f}'+'\t'+f'{limits[ii][1]:.4f}'+'\n')

    integral=np.array(int_tot)

    with open(dir_res+'/'+nameout, 'a') as f:
        f.write('\n')
        for r in range(100):
            f.write('=')
        f.write('\n')
        f.write('\n')
        if not area:
            np.savetxt(dir_res+'/Err.txt', error)
        for j in range(integral.shape[0]):
            if area:
                np.savetxt(dir_res+'/Err_'+str(j+1)+'.txt', error[:,j])
            np.savetxt(dir_res+'/y_'+str(j+1)+'.txt', integral[j,:])
            np.savetxt(dir_res+'/x_'+str(j+1)+'.txt', delays_list)
            f.write('N. interval: '+str(j+1)+'\n')
            f.write('Coefficients\n')
            f.write('A\t'+f'{Coeff[j,0]:.5e}'+'\n')
            f.write('B\t'+f'{Coeff[j,1]:.5e}'+'\n')
            f.write('C\t'+f'{Coeff[j,2]:.5e}'+'\n')
            f.write('D\t'+f'{Coeff[j,3]:.5e}'+'\n')
            f.write('E\t'+f'{Coeff[j,4]:.5e}'+'\n')
            if area:
                f.write('N. point\tIntegral\n')
            else:
                f.write('N. point\tIntensity\n')
            for i in range(len(delays_list)):
                if area:
                    f.write(str(i)+'\t'+f'{integral[j,i]:.3f}'+' +/- '+f'{error[i,j]:.3f}'+'\n')
                else:
                    f.write(str(i)+'\t'+f'{integral[j,i]:.3f}'+' +/- '+f'{error[i]:.3f}'+'\n')
            f.write('\n')

    int_del = np.column_stack((integral.T, delays_list))  #(n. delays x [integral[:,0],...,integral[:,n], delays[:]])
    order = int_del[:,-1].argsort()
    int_del = int_del[order]
    if area:
        error = np.array(error)[order,:]
    else:
        error = np.array(error)[order]
    
    if doexp==True:

        n_peak = 0
        mono_fit = []
        bi_fit = []
        errmono_fit = []
        errbi_fit = []
        fitparameter_f = []
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            if area:
                mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}',err_bar=error[:,ii])
            else:
                mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}',err_bar=error)

            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\nMono\n')
                f.write('y = a + A exp(-t/T1)\nfit: T1=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(mono))
                f.write(report1+'\n')
                f.write('\nBi\n')
                f.write('y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: f=%5.3e, T1a=%5.4e, T1b=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(bi))
                f.write(report2+'\n')
                f.close()

            mono_fit.append(10**tuple(mono)[0])
            errmono_fit.append(err1)
            bi_fit.append((10**tuple(bi)[1], 10**tuple(bi)[2]))
            errbi_fit.append(err2)
            fitparameter_f.append(tuple(bi)[0])
            
            
        with open(dir_res+'/'+'t1.txt', 'w') as f:
            f.write('n.peak\tT1 (s)\terr (s)\tf\n')
            for ii in range(len(mono_fit)):
                try:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+f'{errmono_fit[ii]:.4e}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+'Nan'+'\n')
                try:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+f'{errbi_fit[ii][0]:.4e}'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+f'{errbi_fit[ii][1]:.4e}'+'\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+'Nan'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+'Nan\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
            f.close()
        with open(dir_res+'/'+nameout, 'a') as f:
            for r in range(100):
                f.write('=')
            f.close()

    elif f_int_fit is not None and doexp==False:

        n_peak = 0
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            #f_int_fit must return a dictionary with the fit parameters and the model
            parameters, model = f_int_fit(int_del[:,-1], int_del[:,ii], **fargs, err_bar=error)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('Time (s)')
            if area:
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error[:,ii], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.set_ylabel('Integral')
            else:
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.set_ylabel('Intensity')
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            ax.plot(int_del[:,-1], model, 'r--', lw=0.7)
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Interval_'+str(ii+1)+'.png', dpi=600)
            plt.close()

            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\n')
                f.write(' '.join([f'{key}={val}' for key, val in parameters.items()])+'\n')
                f.close()

    else:

        for ii in range(int_del.shape[-1]-1):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if area:
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error[:,ii], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.set_ylabel('Integral')
            else:
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.set_ylabel('Intensity')
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Interval_'+str(ii+1)+'.png', dpi=600)
            plt.close()

    return dir_res

def model_fit_pseudo2D(path, delays_list, list_path, cal_lim = None, dofit=True, prev_guess=False, prev_fit=None, fast=False, limits1 = None, limits2 = None, L1R = None, L2R = None, err_conf=0.95, doexp=False, f_int_fit=None, fargs=None): 
    
    for i in range(len(delays_list)):
        if 'pdata' not in list_path[i]:
            list_path[i] = list_path[i]+'/pdata/1'
    
    new_dir = os.path.basename(os.path.normpath(path))
    nome_folder = '_modelfit'  
    try:
        os.mkdir(new_dir+nome_folder)
    except:
        ans = input('\nThe directory '+color_term.BOLD+new_dir+nome_folder+color_term.END+' already exists.\nDo you want to:\n[overwrite the existing directory (1)]\ncreate a new directory (2)\ncontinue writing in the existing directory (3)\n>')
        if ans=='1' or ans=='':
            shutil.rmtree(new_dir+nome_folder)           # Removes all the subdirectories!
            os.makedirs(new_dir+nome_folder)
        elif ans=='2':
            new_dir = input('Write new directory name: ')
            os.makedirs(new_dir+nome_folder)
        elif ans=='3':
            pass
        else:
            print('\nINVALID INPUT')
            exit()
    print('Writing directory '+color_term.BOLD+new_dir+nome_folder+color_term.END)

    #creats directorys inside dir_result with the same names as the spectra in path
    dir_result = new_dir+nome_folder
    [os.makedirs(dir_result+'/'+list_path[i][:list_path[idx].index('/pdata')]) for i in range(len(list_path))]
    dir_result_sp = [dir_result+'/'+list_path[i][:list_path[idx].index('/pdata')] for i in range(len(list_path))]  # 'nome_directory_risultato/numero_spettro'

    VCLIST = []
    
    for idx, dir_res in enumerate(dir_result_sp):  #per ogni campo

        print('DIR: ', color_term.CYAN+dir_res+color_term.END)

        nameout = new_dir+'_sp'+list_path[idx][:list_path[idx].index('/pdata')]+'.out'

        delays = delays_list[idx]

        print('PATH TO SPECTRA: ', path+list_path[idx])

        datap, ppm_scale, ngdicp = nmr_spectra_pseudo2d(path+list_path[idx]) #(n.delay x TD)

        title = open(path+list_path[idx][:list_path[idx].index('/pdata')]+'/title').readlines()  # format xxx.xxmT
        if VCLIST==[] or field==True:
            for i,line in enumerate(title):
                if 'The selected field' in line:
                    line = line.replace('\n','')
                    splitline = line.split(' ')
                    for ii in range(len(splitline)):
                        try:
                            field = float(splitline[ii][:-2])
                        except:
                            pass

            VCLIST.append(field)
            field = True

        acqupars = param_acq(ngdicp)
        procpars = param_pr(ngdicp)  #only qsin and em are considered for the apodization

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
        t_aq = np.linspace(0, td*dw, td) + DE

        data = np.array([datap[r,:] for r in range(datap.shape[0]) if sum(datap[r,:]) != 0+1j*0])  #tolgo eventuali spettri non acquisiti
        print('DATA SHAPE: ', data.shape)

        to_order = np.hstack((np.reshape(delays,(len(delays),1)),data))
        to_order = to_order[to_order[:,0].argsort()]    #to_order = to_order[to_order[:,0].argsort()[::-1]]    #to_order = to_order[to_order[:,0].argsort()]    #scelta spettro guess iniziale in base all'ordine

        data = to_order[:,1:]
        delays = to_order[:,0].real

        if cal_lim is not None:
            cal_shift, cal_shift_ppm, data = calibration(ppm_scale, data, cal_lim[0], cal_lim[1]) 
            with open(dir_res+'/'+nameout, 'w') as f:
                f.write('SPECTRA PATH: \n')
                f.write(path+list_path[0]+'\n')
                f.write('\n')
                f.write('CALIBRATION: ('+f'{cal_lim[0]:.5f}'+':'+f'{cal_lim[1]:.5f}'+') ppm\n')
                f.write('in points\n')
                [f.write(str(cal_shift[ii])+'   ') for ii in range(len(cal_shift))]
                f.write('\nin ppm\n')
                [f.write(f'{cal_shift_ppm[ii]:.5f}'+'   ') for ii in range(len(cal_shift_ppm))]
                f.write('\n\n')
                f.write('Points:\n')
                [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
                if VCLIST is not None and len(VCLIST)>0:
                    f.write('\n')
                    f.write('VCLIST point: '+f'{VCLIST[idx]:.2f}'+' T\n')
                f.close()
        else:
            with open(dir_res+'/'+nameout, 'w') as f:
                f.write('SPECTRA PATH: \n')
                f.write(path+list_path[0]+'\n')
                f.write('\n')
                f.write('Points:\n')
                [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
                if VCLIST is not None and len(VCLIST)>0:
                    f.write('\n')
                    f.write('VCLIST point: '+f'{VCLIST[idx]:.2f}'+' T\n')
                f.close()

        if idx == 0:  #gli input vengono generati solo su uno spettro, volendo anche uno per serie
            spettro = data[0,:]
            CI = create_input(ppm_scale, spettro)
            filename1 = input1_gen(CI)
            matrix = read_input_INT(filename1)
            shutil.copy2(filename1, dir_result+'/')
        else:
            if prev_guess:
                pass
            else:
                ans = input('Continue with the previous guess? ([y]|n) ')
                if ans=='y' or ans=='':
                    pass
                elif ans=='n':
                    spettro = data[0,:]
                    CI = create_input(ppm_scale, spettro)
                    filename1 = input("Write new input1 filename: ")
                    CI.write_input_INT(filename1)
                    print('New input1 saved in '+color_term.BOLD+filename1+color_term.END+' in current directory')
                else:
                    print('\nINVALID INPUT')
                    exit()

        matrix = read_input_INT(filename1)
        multiplicity = matrix[:,-1]

        ##INPUT2 GENERATION##
        if idx==0:
            filename2 = input2_gen(CI, matrix, acqupars, procpars)
            shutil.copy2(filename2, dir_result+'/')
        else:
            if prev_guess:
                pass
            else:
                ans = input('Continue with the previous guess? ([y]|n) ')
                if ans=='y' or ans=='':
                    pass
                elif ans=='n':
                    filename2 = input("Write new input2 filename: ")
                    CI.interactive_iguess(matrix[:,:-1], dic, filename2)
                    print('New input2 saved in '+color_term.BOLD+filename2+color_term.END+' in current directory')

        tensore = read_input_INT2(filename2, matrix[:,:-1])

        name_peak = np.zeros(tensore.shape[0], dtype='int64')
        name=0
        prev = 0
        for i in range(tensore.shape[0]):
            if multiplicity[i]==0:
                name += 1
                name_peak[i] = name
                prev=0
            elif multiplicity[i]!=0:
                if prev==multiplicity[i]:
                    name_peak[i] = name
                else:
                    prev = multiplicity[i]
                    name += 1
                    name_peak[i] = name

        with open(dir_res+'/'+nameout, 'a') as f:
            f.write('\nINPUT1: '+filename1+'\n')
            file = open(filename1).readlines()
            f.write('n. peak\t'+file[0])
            [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
            f.write('\nINPUT2: '+filename2+'\n')
            file = open(filename2).readlines()
            f.write('n. peak\t'+file[0])
            [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
            [f.write('=') for r in range(100)]
            f.write('\n')
            [f.write('=') for r in range(100)]
            f.write('\n')
            f.write('(I: fit interval, P: N. point)\n')
            f.close()

        ppm1 = matrix[:,1]
        ppm2 = matrix[:,2]
        ppm1_set = list(set(ppm1))
        ppm2_set = list(set(ppm2))
        ppm1_set.sort()
        ppm2_set.sort()

        tensore = np.append(tensore, np.reshape(multiplicity, (len(multiplicity),1)), axis=1)

        tensor_red_list = []

        shift_tot = []
        integral_tot = []
        error_tot = []
        for i in range(len(ppm1_set)):  #per intervallo

            ppm1 = ppm1_set[i]
            ppm2 = ppm2_set[i]
            print('FIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2))

            tensor_red = np.array([tensore[ii,:] for ii in range(tensore.shape[0]) if tensore[ii,1]==ppm1 and tensore[ii,2]==ppm2])
            tensor_red_list.append(tensor_red)

            param = lmfit.Parameters()
            for j in range(tensor_red.shape[0]):

                if tensor_red[j,0]=='true':
                    param.add('shift_'+str(j+1), value=tensor_red[j,3], min=tensor_red[j,3]-2, max=tensor_red[j,3]+2)
                    param.add('k_'+str(j+1), value=tensor_red[j,4], min=-1, max=1)
                    param.add('lw_'+str(j+1), value=tensor_red[j,5],  min=1e-4, max = 3.5)
                    param.add('ph_'+str(j+1), value=tensor_red[j,6], min=-np.pi, max=np.pi)
                    param.add('xg_'+str(j+1), value=tensor_red[j,7], min=0, max=1)
                else:
                    param.add('shift_'+str(j+1)+'_f', value=tensor_red[j,3], min=tensor_red[j,3]-2, max=tensor_red[j,3]+2)
                    param.add('k_'+str(j+1)+'_f', value=tensor_red[j,4], min=-1, max=1)
                    param.add('lw_'+str(j+1)+'_f', value=tensor_red[j,5],  min=1e-4, max = 3.5)
                    param.add('ph_'+str(j+1)+'_f', value=tensor_red[j,6], min=-np.pi, max=np.pi)
                    param.add('xg_'+str(j+1)+'_f', value=tensor_red[j,7], min=0, max=1)

            param.add('A', value=tensor_red[-1,8])
            param.add('B', value=tensor_red[-1,9])
            param.add('C', value=tensor_red[-1,10]) 
            param.add('D', value=tensor_red[-1,11])
            param.add('E', value=tensor_red[-1,12])
           

            # set limits for the fit
            # limits1 = {'shift':(-0.5,0.5), 'k':(0,1), 'lw':(1e-4,3.5), 'ph':(-np.pi,np.pi), 'xg':(0,1)}
            # the shift is the only parameter whose limits are defined in terms of a delta value to be applied to the initial guess
            if limits1 is not None:
                for key in limits1.keys():
                    if key=='shift':
                        for jj in range(tensor_red.shape[0]):
                            if tensor_red[jj,0]=='true':
                                name_var = key+'_'+str(jj+1)
                            else:
                                name_var = key+'_'+str(jj+1)+'_f'
                            if limits1[key][0] == limits1[key][1]:
                                param[name_var].set(vary=False)
                            else:
                                param[name_var].set(min=param[name_var].value+limits1[key][0], max=param[name_var].value+limits1[key][1])
                    elif key=='A' or key=='B' or key=='C' or key=='D' or key=='E':
                        if limits1[key][0] == limits1[key][1]:
                            param[key].set(vary=False)
                        else:
                            param[key].set(min=limits1[key][0], max=limits1[key][1])
                    else:
                        for jj in range(tensor_red.shape[0]):
                            if tensor_red[jj,0]=='true':
                                name_var = key+'_'+str(jj+1)
                            else:
                                name_var = key+'_'+str(jj+1)+'_f'
                            if limits1[key][0] == limits1[key][1]:
                                param[name_var].set(vary=False)
                            else:
                                param[name_var].set(min=limits1[key][0], max=limits1[key][1])


            param0 = param.copy()
            integral_tot.append([])
            error_tot.append([])
            for j in range(data.shape[0]):  #per delay

                if j>0:
                    param=prev_param.copy()
                    # the second limits are defined not in absolute terms but as a percentage of the initial value
                    # this second dict will be used for the spectra subsequent to the first one (the one used in the definition of the initial guess)
                    # e.g. if shift = (0.9, 1.1) means that the value of the shift is allowed to vary between 0.9*initial and 1.1*initial
                    if limits2 is not None:
                        for key in limits2.keys():
                            if key=='A' or key=='B' or key=='C' or key=='D' or key=='E':
                                if limits2[key][0] == limits2[key][1]:
                                    param[key].set(vary=False)
                                else:
                                    param[key].set(min=param[key].value*limits2[key][0], max=param[key].value*limits2[key][1])
                            else:
                                for jj in range(tensor_red.shape[0]):
                                    if tensor_red[jj,0]=='true':
                                        name_var = key+'_'+str(jj+1)
                                    else:
                                        name_var = key+'_'+str(jj+1)+'_f'
                                    if limits2[key][0] == limits2[key][1]:
                                        param[name_var].set(vary=False)
                                    else:
                                        param[name_var].set(min=param[name_var].value*limits2[key][0], max=param[name_var].value*limits2[key][1])
                else:
                    param=param0.copy()
        
                #### read
                if not dofit:
                    param = lmfit.Parameters()
                    file = open(prev_fit+list_path[idx][:list_path[idx].index('/pdata')]+'/'+new_dir+'popt_sp'+str(idx)+'_I'+str(i)+'_P'+str(j), 'r')
                    param.load(file)
                ###
            
                peak_int, int_err, prev_param_compl, result, _ = fit_peaks_bsl_I(param, ppm_scale, data[j,:], tensor_red, 
                                                                t_aq, sf1, o1p, td, dw, j, i, dir_res, new_dir, SR=SR, 
                                                                SI=SI, SW=SW, LB=LB, SSB=SSB, dofit=dofit,fast=fast, L1R=L1R, 
                                                                L2R=L2R, err_conf=err_conf)
                
                #### write
                if dofit:
                    file = open(dir_res+'/'+new_dir+'popt_sp'+str(idx)+'_I'+str(i)+'_P'+str(j), 'w')
                    prev_param_compl.dump(file)
                    if prev_fit is not None:
                        file = open(prev_fit+new_dir+'popt_sp'+str(idx)+'_I'+str(i)+'_P'+str(j), 'w')
                        prev_param_compl.dump(file)
                ####

                prev_param = prev_param_compl.copy()

                integral_tot[-1].append(peak_int)
                error_tot[-1].append(int_err)

                peaks_shift = [prev_param_compl['shift_'+str(jj+1)].value for jj in range(tensor_red.shape[0]) if tensor_red[jj,0]=='true']
                
                with open(dir_res+'/'+nameout, 'a') as f:
                    f.write('\n')
                    [f.write('-') for r in range(30)]
                    f.write('\nFIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2)+'\n')
                    f.write('I: '+str(i+1)+' P: '+str(j+1)+'\n')
                    [f.write('-') for r in range(30)]
                    f.write('\n')
                    f.write('\nFit Report: \n')
                    f.write(lmfit.fit_report(result)+'\n')
                    f.write('\nn.peak\tShift\tIntegral\n')
                    for jj in range(len(peak_int)):
                        f.write(f'{jj+1}    {peaks_shift[jj]:.3f}    {peak_int[jj]:.3f} +/- {np.abs(int_err[jj]):.3f}\n')
                    f.close()

            integral_tot[-1]=np.array(integral_tot[-1])
            error_tot[-1]=np.array(error_tot[-1])

        integral=np.concatenate(integral_tot, axis=1)
        error = np.concatenate(error_tot, axis=1) 

        for j in range(integral.shape[1]):
            np.savetxt(dir_res+'/Err_'+str(j+1)+'.txt', error[:,j])
            np.savetxt(dir_res+'/y_'+str(j+1)+'.txt', integral[:,j])
            np.savetxt(dir_res+'/x_'+str(j+1)+'.txt', delays)

        int_del = np.column_stack((integral, delays))  #(n. delays x [integral[:,0],...,integral[:,n], delays[:]])
        order = int_del[:,-1].argsort()
        int_del = int_del[order]
        try:
            error = np.reshape(error[order], (len(error),))
        except:
            error = np.reshape(error[order], (len(error),error.shape[-1]))
        
        if doexp==True:

            n_peak = 0
            mono_fit = []
            bi_fit = []
            errmono_fit = []
            errbi_fit = []
            fitparameter_f = []
            for ii in range(int_del.shape[-1]-1):
                n_peak += 1
                mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}', err_bar=error[:,ii])
                
                with open(dir_res+'/'+nameout, 'a') as f:
                    f.write('\n')
                    [f.write('-') for r in range(30)]
                    f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                    [f.write('-') for r in range(30)]
                    f.write('\n')
                    f.write('\nFit Parameters:\n')
                    f.write('\nMono\n')
                    f.write('y = a + A exp(-t/T1)\nfit: T1=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(mono))
                    f.write(report1+'\n')
                    f.write('\nBi\n')
                    f.write('y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: f=%5.3e, T1a=%5.4e, T1b=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(bi))
                    f.write(report2+'\n')
                    f.close()

                mono_fit.append(10**tuple(mono)[0])
                errmono_fit.append(err1)
                bi_fit.append((10**tuple(bi)[1], 10**tuple(bi)[2]))
                errbi_fit.append(err2)
                fitparameter_f.append(tuple(bi)[0])
                
                
            with open(dir_res+'/'+'t1.txt', 'w') as f:
                f.write('n.peak\tT1 (s)\terr (s)\tf\n')
                for ii in range(len(mono_fit)):
                    try:
                        f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+f'{errmono_fit[ii]:.4e}'+'\n')
                    except:
                        f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+'Nan'+'\n')
                    try:
                        f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+f'{errbi_fit[ii][0]:.4e}'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+f'{errbi_fit[ii][1]:.4e}'+'\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                    except:
                        f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+'Nan'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+'Nan\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                f.close()
            with open(dir_res+'/'+nameout, 'a') as f:
                for r in range(100):
                    f.write('=')
                f.close()

        elif f_int_fit is not None and doexp==False:

            n_peak = 0
            for ii in range(int_del.shape[-1]-1):
                n_peak += 1
                #f_int_fit must return a dictionary with the fit parameters and the model
                parameters, model = f_int_fit(int_del[:,-1], int_del[:,ii], **fargs, err_bar=error)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
                ax.plot(int_del[:,-1], model, 'r--', lw=0.7)
                #ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
                plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
                plt.close()

                with open(dir_res+'/'+nameout, 'a') as f:
                    f.write('\n')
                    [f.write('-') for r in range(30)]
                    f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                    [f.write('-') for r in range(30)]
                    f.write('\n')
                    f.write('\nFit Parameters:\n')
                    f.write('\n')
                    f.write(' '.join([f'{key}={val}' for key, val in parameters.items()])+'\n')
                    f.close()

        else:

            for ii in range(int_del.shape[-1]-1):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
                ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
                #ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
                plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
                plt.close()

        
    VCLIST = np.array(VCLIST)
    np.savetxt(dir_result+'/VCLIST.txt', VCLIST)

    return dir_result

def model_fit_1D(path, delays_list, list_path, cal_lim = None, dofit=True, prev_fit=None, fast=False, limits1 = None, limits2 = None, L1R = None, L2R = None, err_conf=0.95, doexp=False, f_int_fit=None, fargs=None):    

    for i in range(len(delays_list)):
        if 'pdata' not in list_path[i]:
            list_path[i] = list_path[i]+'/pdata/1'
    
    new_dir = os.path.basename(os.path.normpath(path))
    nome_folder = '_modelfit'   
    try:
        os.mkdir(new_dir+nome_folder)
    except:
        ans = input('\nThe directory '+color_term.BOLD+new_dir+nome_folder+color_term.END+' already exists.\nDo you want to:\n[overwrite the existing directory (1)]\ncreate a new directory (2)\ncontinue writing in the existing directory (3)\n>')
        if ans=='1' or ans == '':
            shutil.rmtree(new_dir+nome_folder)           # Removes all the subdirectories!
            os.makedirs(new_dir+nome_folder)
        elif ans=='2':
            new_dir = input('Write new directory name: ')
            os.makedirs(new_dir+nome_folder)
        elif ans=='3':
            pass
        else:
            print('\nINVALID INPUT')
            exit()
    print('Writing directory '+color_term.BOLD+new_dir+nome_folder+color_term.END)

    dir_res = new_dir+nome_folder
    print('DIR: ', color_term.CYAN+dir_res+color_term.END)

    nameout = new_dir+'.out'

    print('PATH TO SPECTRA: ', path+'/'+dir_res)

    data_d = []
    ppmscale_d = []
    for i in range(len(delays_list)):
        data, ppmscale, ngdicp = nmr_spectra_1d(path+list_path[i])
        data_d.append(data)
        ppmscale_d.append(ppmscale)

    data = np.array(data_d)
    ppm_scale = ppmscale.copy()
    
    acqupars = param_acq(ngdicp)
    procpars = param_pr(ngdicp)  #only qsin and em are considered for the apodization

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
    t_aq = np.linspace(0, td*dw, td) + DE

    to_order = np.hstack((np.reshape(delays_list,(len(delays_list),1)),data))

    to_order = to_order[to_order[:,0].argsort()[::-1]]	

    data = to_order[:,1:]
    delays = to_order[:,0].real

    if cal_lim is not None:
        cal_shift, cal_shift_ppm, data = calibration(ppm_scale, data, cal_lim[0], cal_lim[1]) 
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('CALIBRATION: ('+f'{cal_lim[0]:.5f}'+':'+f'{cal_lim[1]:.5f}'+') ppm\n')
            f.write('in points\n')
            [f.write(str(cal_shift[ii])+'   ') for ii in range(len(cal_shift))]
            f.write('\nin ppm\n')
            [f.write(f'{cal_shift_ppm[ii]:.5f}'+'   ') for ii in range(len(cal_shift_ppm))]
            f.write('\n\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
            f.close()
    else:
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
            f.close()

    spettro = data[0,:]  #spectra initial guess

    CI = create_input(ppm_scale, spettro)
    filename1 = input1_gen(CI)
    matrix = read_input_INT(filename1)
    shutil.copy2(filename1, dir_res+'/')
    matrix = read_input_INT(filename1)
    multiplicity = matrix[:,-1]

    ##INPUT2 GENERATION##
    filename2 = input2_gen(CI, matrix, acqupars, procpars)
    shutil.copy2(filename2, dir_res+'/')
    tensore = read_input_INT2(filename2, matrix[:,:-1])

    name_peak = np.zeros(tensore.shape[0], dtype='int64')
    name=0
    prev = 0
    for i in range(tensore.shape[0]):
        if multiplicity[i]==0:
            name += 1
            name_peak[i] = name
            prev=0
        elif multiplicity[i]!=0:
            if prev==multiplicity[i]:
                name_peak[i] = name
            else:
                prev = multiplicity[i]
                name += 1
                name_peak[i] = name

    with open(dir_res+'/'+nameout, 'a') as f:
        f.write('\nINPUT1: '+filename1+'\n')
        file = open(filename1).readlines()
        f.write('n. peak\t'+file[0])
        [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
        f.write('\nINPUT2: '+filename2+'\n')
        file = open(filename2).readlines()
        f.write('n. peak\t'+file[0])
        [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
        [f.write('=') for r in range(100)]
        f.write('\n')
        [f.write('=') for r in range(100)]
        f.write('\n')
        f.write('(I: fit interval, P: N. point)\n')
        f.close()

    ppm1 = matrix[:,1]
    ppm2 = matrix[:,2]
    ppm1_set = list(set(ppm1))
    ppm2_set = list(set(ppm2))
    ppm1_set.sort()
    ppm2_set.sort()

    tensore = np.append(tensore, np.reshape(multiplicity, (len(multiplicity),1)), axis=1)

    tensor_red_list = []

    shift_tot = []
    integral_tot = []
    error_tot = []
    for i in range(len(ppm1_set)):  #per intervallo

        ppm1 = ppm1_set[i]
        ppm2 = ppm2_set[i]
        print('FIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2))

        tensor_red = np.array([tensore[ii,:] for ii in range(tensore.shape[0]) if tensore[ii,1]==ppm1 and tensore[ii,2]==ppm2])
        tensor_red_list.append(tensor_red)

        param = lmfit.Parameters()
        for j in range(tensor_red.shape[0]):

            if tensor_red[j,0]=='true':
                param.add('shift_'+str(j+1), value=tensor_red[j,3], min=tensor_red[j,3]-2, max=tensor_red[j,3]+2)
                param.add('k_'+str(j+1), value=tensor_red[j,4], min=-1, max=1)
                param.add('lw_'+str(j+1), value=tensor_red[j,5],  min=1e-4, max = 3.5)
                param.add('ph_'+str(j+1), value=tensor_red[j,6], min=-np.pi, max=np.pi)
                param.add('xg_'+str(j+1), value=tensor_red[j,7], min=0, max=1)
            else:
                param.add('shift_'+str(j+1)+'_f', value=tensor_red[j,3], min=tensor_red[j,3]-2, max=tensor_red[j,3]+2)
                param.add('k_'+str(j+1)+'_f', value=tensor_red[j,4], min=-1, max=1)
                param.add('lw_'+str(j+1)+'_f', value=tensor_red[j,5],  min=1e-4, max = 3.5)
                param.add('ph_'+str(j+1)+'_f', value=tensor_red[j,6], min=-np.pi, max=np.pi)
                param.add('xg_'+str(j+1)+'_f', value=tensor_red[j,7], min=0, max=1)

        param.add('A', value=tensor_red[-1,8])
        param.add('B', value=tensor_red[-1,9])
        param.add('C', value=tensor_red[-1,10]) 
        param.add('D', value=tensor_red[-1,11])
        param.add('E', value=tensor_red[-1,12])

        # set limits for the fit
        # limits1 = {'shift':(-0.5,0.5), 'k':(0,1), 'lw':(1e-4,3.5), 'ph':(-np.pi,np.pi), 'xg':(0,1)}
        # the shift is the only parameter whose limits are defined in terms of a delta value to be applied to the initial guess
        if limits1 is not None:
            for key in limits1.keys():
                if key=='shift':
                    for jj in range(tensor_red.shape[0]):
                        if tensor_red[jj,0]=='true':
                            name_var = key+'_'+str(jj+1)
                        else:
                            name_var = key+'_'+str(jj+1)+'_f'
                        if limits1[key][0] == limits1[key][1]:
                            param[name_var].set(vary=False)
                        else:
                            param[name_var].set(min=param[name_var].value+limits1[key][0], max=param[name_var].value+limits1[key][1])
                elif key=='A' or key=='B' or key=='C' or key=='D' or key=='E':
                    if limits1[key][0] == limits1[key][1]:
                        param[key].set(vary=False)
                    else:
                        param[key].set(min=limits1[key][0], max=limits1[key][1])
                else:
                    for jj in range(tensor_red.shape[0]):
                        if tensor_red[jj,0]=='true':
                            name_var = key+'_'+str(jj+1)
                        else:
                            name_var = key+'_'+str(jj+1)+'_f'
                        if limits1[key][0] == limits1[key][1]:
                            param[name_var].set(vary=False)
                        else:
                            param[name_var].set(min=limits1[key][0], max=limits1[key][1])

        integral_tot.append([])
        error_tot.append([])

        param0 = param.copy()
        for j in range(data.shape[0]):  #per delay

            if j>0:
                param=prev_param.copy()
                # the second limits are defined not in absolute terms but as a percentage of the initial value
                # this second dict will be used for the spectra subsequent to the first one (the one used in the definition of the initial guess)
                # e.g. if shift = (0.9, 1.1) means that the value of the shift is allowed to vary between 0.9*initial and 1.1*initial
                if limits2 is not None:
                    for key in limits2.keys():
                        if key=='A' or key=='B' or key=='C' or key=='D' or key=='E':
                            if limits2[key][0] == limits2[key][1]:
                                param[key].set(vary=False)
                            else:
                                param[key].set(min=param[key].value*limits2[key][0], max=param[key].value*limits2[key][1])
                        else:
                            for jj in range(tensor_red.shape[0]):
                                if tensor_red[jj,0]=='true':
                                    name_var = key+'_'+str(jj+1)
                                else:
                                    name_var = key+'_'+str(jj+1)+'_f'
                                if limits2[key][0] == limits2[key][1]:
                                    param[name_var].set(vary=False)
                                else:
                                    param[name_var].set(min=param[name_var].value*limits2[key][0], max=param[name_var].value*limits2[key][1])
            else:
                param=param0.copy()

            #### read
            if not dofit:
                param = lmfit.Parameters()
                file = open(prev_fit+new_dir+'popt_I'+str(i)+'_P'+str(j), 'r')
                param.load(file)
            ###
          
            peak_int, int_err, prev_param_compl, result, _ = fit_peaks_bsl_I(param, ppm_scale, data[j,:], tensor_red, 
                                                            t_aq, sf1, o1p, td, dw, j, i, dir_res, new_dir, SR=SR, 
                                                            SI=SI, SW=SW, LB=LB, SSB=SSB, dofit=dofit, fast=fast, L1R=L1R, 
                                                            L2R=L2R, err_conf=err_conf)
            
            #### write
            if dofit:
                file = open(dir_res+'/'+new_dir+'popt_I'+str(i)+'_P'+str(j), 'w')
                prev_param_compl.dump(file)
                if prev_fit is not None:
                    file = open(prev_fit+new_dir+'popt_I'+str(i)+'_P'+str(j), 'w')
                    prev_param_compl.dump(file)
            ####

            prev_param = prev_param_compl.copy()

            integral_tot[-1].append(peak_int)
            error_tot[-1].append(int_err)

            peaks_shift = [prev_param_compl['shift_'+str(jj+1)].value for jj in range(tensor_red.shape[0]) if tensor_red[jj,0]=='true']
            
            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nFIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2)+'\n')
                f.write('I: '+str(i+1)+' P: '+str(j+1)+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Report: \n')
                f.write(lmfit.fit_report(result)+'\n')
                f.write('\nn.peak\tShift\tIntegral\n')
                for jj in range(len(peak_int)):
                    f.write(f'{jj+1}    {peaks_shift[jj]:.3f}    {peak_int[jj]:.3f} +/- {np.abs(int_err[jj]):.3f}\n')
                f.close()
                
    integral=np.concatenate(integral_tot, axis=1)
    error = np.concatenate(error_tot, axis=1) 

    for j in range(integral.shape[1]):
        np.savetxt(dir_res+'/Err_'+str(j+1)+'.txt', error[:,j])
        np.savetxt(dir_res+'/y_'+str(j+1)+'.txt', integral[:,j])
        np.savetxt(dir_res+'/x_'+str(j+1)+'.txt', delays)

    int_del = np.column_stack((integral, delays))  #(n. delays x [integral[:,0],...,integral[:,n], delays[:]])
    order = int_del[:,-1].argsort()
    int_del = int_del[order]
    try:
        error = np.reshape(error[order], (len(error),))
    except:
        error = np.reshape(error[order], (len(error),error.shape[-1]))
        error = error[:,0]
    
    if doexp==True:

        n_peak = 0
        mono_fit = []
        bi_fit = []
        errmono_fit = []
        errbi_fit = []
        fitparameter_f = []
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}', err_bar=error)
            
            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\nMono\n')
                f.write('y = a + A exp(-t/T1)\nfit: T1=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(mono))
                f.write(report1+'\n')
                f.write('\nBi\n')
                f.write('y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: f=%5.3e, T1a=%5.4e, T1b=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(bi))
                f.write(report2+'\n')
                f.close()

            mono_fit.append(10**tuple(mono)[0])
            errmono_fit.append(err1)
            bi_fit.append((10**tuple(bi)[1], 10**tuple(bi)[2]))
            errbi_fit.append(err2)
            fitparameter_f.append(tuple(bi)[0])
            
            
        with open(dir_res+'/'+'t1.txt', 'w') as f:
            f.write('n.peak\tT1 (s)\terr (s)\tf\n')
            for ii in range(len(mono_fit)):
                try:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+f'{errmono_fit[ii]:.4e}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+'Nan'+'\n')
                try:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+f'{errbi_fit[ii][0]:.4e}'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+f'{errbi_fit[ii][1]:.4e}'+'\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+'Nan'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+'Nan\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
            f.close()
        with open(dir_res+'/'+nameout, 'a') as f:
            for r in range(100):
                f.write('=')
            f.close()

    elif f_int_fit is not None and doexp==False:

        n_peak = 0
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            #f_int_fit must return a dictionary with the fit parameters and the model
            parameters, model = f_int_fit(int_del[:,-1], int_del[:,ii], **fargs, err_bar=error)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            ax.plot(int_del[:,-1], model, 'r--', lw=0.7)
            
            #ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
            plt.close()

            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\n')
                f.write(' '.join([f'{key}={val}' for key, val in parameters.items()])+'\n')
                f.close()

    else:

        for ii in range(int_del.shape[-1]-1):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            #ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
            plt.close()


    return dir_res

def fit_peaks_bsl_I(param, ppm_scale, spettro, tensor_red, t_aq, sf1, o1p, td, dw, j, jj, dir_res, new_dir, SR=0, SI=0, SW=0, LB=0, SSB=0, dofit=True, fast=False, IR=False, L1R=None, L2R=None, err_conf=0.95):
    
    cal = SR/sf1 - (ppm_scale[0]-ppm_scale[1])

    #spettro = spettro0/np.max(spettro0.real)
    
    cycle = -1   
    def f_residue(param, ppm_scale, spettro, tensor_red, result=False):
        nonlocal cycle
        cycle += 1
        print('fit peaks: '+f'{cycle:5g}',end='\r')
        par = param.valuesdict()

        lor_list = []
        lor_ph0_list = []
        comp_list = []
        prev=0
        sim_spectra_tot = np.zeros_like(spettro, dtype='float64')

        mult = tensor_red[:,-1]

        sx, dx, zero = find_limits(tensor_red[0,1], tensor_red[0,2], ppm_scale)

        sim_spectra = np.zeros_like(spettro, dtype='float64')
        
        for ii in range(tensor_red.shape[0]):  #per ogni voigt

            try:
                lor = t_voigt(t_aq, (par['shift_'+str(ii+1)]+cal-o1p)*sf1, 2*np.pi*par['lw_'+str(ii+1)]*sf1,
                                            A=par['k_'+str(ii+1)], phi=par['ph_'+str(ii+1)], x_g=par['xg_'+str(ii+1)])
            except:
                lor = t_voigt(t_aq, (par['shift_'+str(ii+1)+'_f']+cal-o1p)*sf1, 2*np.pi*par['lw_'+str(ii+1)+'_f']*sf1,
                                            A=par['k_'+str(ii+1)+'_f'], phi=par['ph_'+str(ii+1)+'_f'], x_g=par['xg_'+str(ii+1)+'_f'])
            
            ### processing
            lor *= em(lor, LB, SW)
            lor *= qsin(lor, SSB)
            lor = zf(lor, SI)
            ###

            #mi serve per l'errore 
            if result:
                try:
                    lor_ph0 = t_voigt(t_aq, (par['shift_'+str(ii+1)]+cal-o1p)*sf1, 2*np.pi*par['lw_'+str(ii+1)]*sf1,
                                                A=np.abs(par['k_'+str(ii+1)]), phi=0, x_g=par['xg_'+str(ii+1)])
                    ### processing
                    lor_ph0 = zf(lor_ph0, SI)
                    ###
                except:
                    lor_ph0 = None


            lor = ft(lor, SI, dw, o1p, sf1)[0]
            lor = np.conj(lor)[::-1].real

            if np.isnan(lor).any():
                print('NAN')
                print(ii, par['shift_'+str(ii+1)], par['lw_'+str(ii+1)], par['k_'+str(ii+1)], par['ph_'+str(ii+1)], par['xg_'+str(ii+1)])


            sim_spectra += lor
            comp_list.append(lor[sx:dx])
            
            if result:
                if lor_ph0 is not None:
                    lor_ph0 = ft(lor_ph0, SI, dw, o1p, sf1)[0]
                    lor_ph0 = np.conj(lor_ph0)[::-1].real

                    m=mult[ii]
                    if m==0:
                        prev=0
                        lor_list.append(lor)
                        lor_ph0_list.append(lor_ph0)
                    elif m!=0:
                        if m==prev:
                            lor_m = lor+lor_list[-1]
                            lor_m_int = lor_ph0+lor_ph0_list[-1]
                            del lor_list[-1]
                            lor_list.append(lor_m)
                            del lor_ph0_list[-1]
                            lor_ph0_list.append(lor_m_int)
                        else:
                            prev = m
                            lor_list.append(lor)
                            lor_ph0_list.append(lor_ph0)

            sim_spectra_tot += sim_spectra

        x = ppm_scale[sx:dx]-zero
        corr_baseline = par['E']*x**4 + par['D']*x**3 + par['C']*x**2 + par['B']*x + par['A']

        cost = np.max(spettro.real)  #useless since it is normaliazed
        
        if result:
            lor_ph0_list = np.array(lor_ph0_list)*cost  
            lor_list = np.array(lor_list)*cost
        comp_list = np.array(comp_list)*cost

        model = cost*(corr_baseline+sim_spectra[sx:dx]) 

        res = model.real-spettro[sx:dx].real

        if L1R is not None:
            sum_param = np.sum(np.array([np.abs(par[key]) for key in par.keys()]))/np.max(spettro.real)
            res += L1R*sum_param
        elif L2R is not None:
            sum_param = np.sum(np.array([par[key]**2 for key in par.keys()]))/np.max(spettro.real)
            res += L2R*sum_param
        else:
            pass

        if cycle%1000==0 or result:
            f_figure_comp(ppm_scale[sx:dx], spettro[sx:dx], model, comp_list, 
                                name=dir_res+'/'+new_dir+'_P'+str(j+1)+'_I'+str(jj+1), basefig = cost*corr_baseline, 
                                dic_fig={'h':5.59,'w':4.56, 'sx':tensor_red[0,1], 'dx':tensor_red[0,2]})
            histogram(res, nbins=100, density=True, f_lims= None, xlabel=None, x_symm=False, name=dir_res+'/'+new_dir+'_P'+str(j+1)+'_I'+str(jj+1)+'_hist')

        if result:
            integral_in=[]
            int_err = []
            sx, dx, zero = find_limits(tensor_red[0,1], tensor_red[0,2], ppm_scale)
            for ii in range(len(lor_list)):
                if lor_ph0_list[ii] is not None:
                    if IR:
                        integral_in.append(np.trapz(lor_list[ii]))
                    else:
                        integral_in.append(np.trapz(lor_ph0_list[ii]))
                    Err = error_calc_num(ppm_scale, res, lor_ph0_list[ii], np.trapz(lor_ph0_list[ii]), sx, dx, confidence=err_conf)
                    int_err.append(Err)

        if not result:
            residuals = res
            x = ppm_scale
            corr_baseline = par['E']*x**4 + par['D']*x**3 + par['C']*x**2 + par['B']*x + par['A']

            if np.isnan(residuals).any():
                print('NAN')
                pprint(par)
                print(cost)
                print(np.isnan(model).any(), np.isnan(spettro).any())
                print(np.isnan(sim_spectra).any(), np.isnan(corr_baseline).any())
                exit()

            return residuals
        else:

            x = ppm_scale[sx:dx]-zero
            corr_baseline = par['E']*x**4 + par['D']*x**3 + par['C']*x**2 + par['B']*x + par['A']
            model = cost*(corr_baseline+sim_spectra_tot[sx:dx])
            #f_fun.f_figure(ppm_scale[sx:dx], spettro[sx:dx], model, name=dir_res+'/'+new_dir+'_D'+str(j+1)+'_I'+str(jj+1), basefig = cost*corr_baseline)

            return integral_in, int_err, sim_spectra, spettro[sx:dx]-corr_baseline*cost
        
   
    minner = lmfit.Minimizer(f_residue, param, fcn_args=(ppm_scale, spettro, tensor_red))
    if dofit:
        if not fast:
            result = minner.minimize(method='Nelder', max_nfev=10000)#, xtol=1e-7, ftol=1e-7)
            params = result.params
            result = minner.minimize(params=params, method='leastsq', max_nfev=10000)
        else:
            result = minner.minimize(method='leastsq', max_nfev=10000)
    else:
        result = minner.minimize(method='Nelder', max_nfev=0)
    
    popt = result.params
    peak_int, int_err, sim_spectra, spettro_corrbsl = f_residue(popt, ppm_scale, spettro, tensor_red, result=True)
    
    return peak_int, int_err, popt, result, spettro_corrbsl

#------------------------------------------------------------#
#                       TAILORED FUNCTIONS                   #
#------------------------------------------------------------#

def fit_exponential(x, y, name= None, err_bar=None, figura=True):

    def exponential(param, x, y, result=False):
        par=param.valuesdict()
        t1 = 10**par['t1']
        model = np.exp(-x/t1)
        den = np.mean(model**2)-np.mean(model)**2
        a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
        A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
        model *= A
        model += a
        if result==False:
            return y-model
        else:
            return model, a, A

    def bi_exponential(param, x, y, result=False):
        par=param.valuesdict()
        f = par['f']
        t1a = 10**par['t1a']
        t1b = 10**par['t1b']
        model = (f*np.exp(-x/t1a)+(1-f)*np.exp(-x/t1b))
        den = np.mean(model**2)-np.mean(model)**2
        a = (np.mean(model**2)*np.mean(y)-np.mean(model*y)*np.mean(model))/den
        A = (np.mean(model*y)-(np.mean(model)*np.mean(y)))/den
        model *= A
        model += a
        if result==False:
            return y-model
        else:
            return model, a, A

    param = lmfit.Parameters()
    param.add('t1', value=0, min=-4, max=1)
    minner = lmfit.Minimizer(exponential, param, fcn_args=(x, y))
    result1 = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-8, ftol=1e-8)
    popt1 = result1.params
    report1 = lmfit.fit_report(result1)

    param = lmfit.Parameters()
    param.add('f', value=0.5, min=0, max=1)
    param.add('t1a', value=0, min=-4, max=1)
    param.add('t1b', value=0, min=-4, max=1)
    minner = lmfit.Minimizer(bi_exponential, param, fcn_args=(x, y))
    result2 = minner.minimize(method='leastsq', max_nfev=10000, xtol=1e-8, ftol=1e-8)
    popt2 = result2.params
    report2 = lmfit.fit_report(result2)

    func1, a, A = exponential(popt1, x, y, result=True)
    popt1.add('a', value=a)
    popt1.add('A', value=A)
    func2, a, A = bi_exponential(popt2, x, y, result=True)
    popt2.add('a', value=a)
    popt2.add('A', value=A)

    fig=plt.figure()
    fig.set_size_inches(3.59,2.56)
    plt.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.80)
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(labelsize=5)
    ax.plot(x, y, 'o', c='k', markersize=0.7)
    ax.set_xscale('log')
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
    label_mod1 = ''
    label_mod2 = ''
    for key, values in popt2.items():
        if key=='t1a':
            try:
                label_mod1+=f'{10**values.value:.5e}'+r' $\pm$ '+f'{np.abs(10**values.value*values.stderr/values.value):.5e}'
                err21 = np.abs(10**values.value*values.stderr/values.value)
            except:
                label_mod1+=f'{10**values.value:.5e}'+r' $\pm$ Nan'
                err21 = None
        elif key=='t1b':
            try:
                label_mod2+=f'{10**values.value:.5e}'+r' $\pm$ '+f'{np.abs(10**values.value*values.stderr/values.value):.5e}'
                err22 = np.abs(10**values.value*values.stderr/values.value)
            except:
                label_mod2+=f'{10**values.value:.5e}'+r' $\pm$ Nan'
                err22 = None

    ax.plot(x, func1, 'r--', lw=0.7, label='y = a + A exp(-t/T1)\nfit: T1 = '+label_mod+' sec\n'+r'$\chi^2$ red: '+f'{result1.redchi:.5e}')
    ax.plot(x, func2, 'g--', lw=0.7, label='y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: T1a = '+label_mod1+' sec, T1b = '+label_mod2+' sec\n'+r'$\chi^2$ red: '+f'{result2.redchi:.5e}')
    ax.set_xlabel('Delay (s)', fontsize=6)
    ax.set_ylabel('Intensity (a.u.)', fontsize=6)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3,3), useMathText=True)
    ax.yaxis.get_offset_text().set_size(5)
    ax.legend(fontsize=3.5, bbox_to_anchor=(0.95,0.98), bbox_transform=fig.transFigure)
    if figura:
        if name is None:
            plt.show()
        else:
            plt.savefig(name+'.png', dpi=600)
            plt.close()

        return popt1.valuesdict().values(), popt2.valuesdict().values(), report1, report2, err1, (err21, err22)
    else:
        plt.close()
        return popt1.valuesdict().values(), popt2.valuesdict().values(), func1, func2, err1, (err21, err22)  #for theUltimatePlot

def theUltimatePlot(dir_result, list_path, bi_list=None, colormap = 'hsv', area=False, VClist=None):
    
    if VClist is None:
        VClist = np.loadtxt(dir_result+'/VCLIST.txt')

    hsv = plt.get_cmap(colormap)
    hsv_colors = [hsv(i / len(VClist)) for i in range(len(VClist))]
    
    mono_tot = []  # n. campi x n.peaks x ...
    bi_tot = []
    x_tot = []   # n.campi x n.peaks x (delay, intensity)
    y_tot = []
    yerr_tot = []
    for idx in range(len(list_path)):
        
        directory = dir_result+'/'+list_path[idx][:list_path[idx].index('/pdata')]+'/'+'t1.txt'
        file = open(directory).readlines()
        mono = []
        bi = []
        for i,line in enumerate(file):
            splitline = line.split('\t')
            if i>0 and i%2!=0:
                mono.append([float(el) for el in splitline])
            elif i>0 and i%2==0:
                bi.append([float(el) for el in splitline])
                
        mono_tot.append(mono)
        bi_tot.append(bi)
        
        x_tot.append([])
        y_tot.append([])
        yerr_tot.append([])
        for i in range(len(mono)):
            x_tot[idx].append(np.loadtxt(dir_result+'/'+list_path[idx][:list_path[idx].index('/pdata')]+'/x_'+str(i+1)+'.txt'))
            y_tot[idx].append(np.loadtxt(dir_result+'/'+list_path[idx][:list_path[idx].index('/pdata')]+'/y_'+str(i+1)+'.txt'))
            if area:
                yerr_tot[idx].append(np.loadtxt(dir_result+'/'+list_path[idx][:list_path[idx].index('/pdata')]+'/Err_'+str(i+1)+'.txt'))
            else:
                yerr_tot[idx].append(np.loadtxt(dir_result+'/'+list_path[idx][:list_path[idx].index('/pdata')]+'/Err.txt'))
        x_tot[-1] = np.array(x_tot[-1])
        y_tot[-1] = np.array(y_tot[-1])
        yerr_tot[-1] = np.array(yerr_tot[-1])
        
    mono_tot = np.array(mono_tot)  
    bi_tot = np.array(bi_tot)
    x_tot = np.array(x_tot)
    y_tot = np.array(y_tot)
    yerr_tot = np.array(yerr_tot)

    for i in range(mono_tot.shape[1]):
        
        x = x_tot[:,i,:]
        y = y_tot[:,i,:]
        yerr = yerr_tot[:,i,:]
        
        if bi_list is None:
            biflag = False
        else:
            if i+1 in bi_list:
                biflag = True
            else:
                biflag = False
        
        fig=plt.figure()
        fig.set_size_inches(5,3)
        plt.subplots_adjust(left=0.10,bottom=0.10,right=0.80,top=0.95)
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('Delay (s)', fontsize=6)
        ax.set_ylabel('Intensity (a.u.)', fontsize=6)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3,3), useMathText=True)
        ax.yaxis.get_offset_text().set_size(5)
        ax.set_xscale('log')
        R1 = []
        err = []
        for ii in range(len(VClist)):
            monopar, bipar, func1, func2, err1, err2 = fit_exponential(x[ii], y[ii], figura=False, err_bar=yerr[ii])
            monopar = list(monopar)
            bipar = list(bipar)
            line, = ax.plot(x[ii], y[ii], 'o', markersize=0.7, label = f'{VClist[ii]:.0f}'+' mT', c=hsv_colors[ii])
            ax.errorbar(x[ii], y[ii], yerr=yerr[ii], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            if biflag:
                ax.plot(x[ii], func2, '--', lw=0.7, c=line.get_color(), label='T1a = '+f'{10**bipar[1]:.4e}'+' s T1b = '+f'{10**bipar[2]:.4e}'+' s\n'+'f = '+f'{bipar[0]:.3e}')
                R1.append([1/10**bipar[1], 1/10**bipar[2]])
                err_bi = [0, 1]
                if err2[0] is None or (err2[0]/10**bipar[1]*R1[-1][0])>R1[-1][0]/2:
                    err_bi[0] = np.nan
                else:
                    err_bi[0] = err2[0]/10**bipar[1]*R1[-1][0]
                if err2[1] is None or (err2[1]/10**bipar[2]*R1[-1][1])>R1[-1][1]/2:
                    err_bi[1] = np.nan
                else:
                    err_bi[1] = err2[1]/10**bipar[2]*R1[-1][1]
                err.append(err_bi)
            else:
                ax.plot(x[ii], func1, '--', lw=0.7, c=line.get_color(), label='T1 = '+f'{10**monopar[0]:.4e}'+' s')
                R1.append(1/10**monopar[0])
                if err1 is None or (err1/10**monopar[0]*R1[-1])>R1[-1]/2:
                    err.append(np.nan)
                else:
                    err.append(err1/10**monopar[0]*R1[-1])
        
        ax.legend(fontsize=3, bbox_to_anchor=(0.9,0.95), bbox_transform=fig.transFigure)
        plt.savefig(dir_result+'/'+str(i+1)+'.png', dpi=600)
        plt.close()

        R1 = np.array(R1)
        err = np.array(err)

        fig=plt.figure()
        fig.set_size_inches(5,3)
        plt.subplots_adjust(left=0.10,bottom=0.10,right=0.80,top=0.95)
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('Field (mT)', fontsize=6)
        ax.set_ylabel(r'$R_1$ ($s^{-1}$)', fontsize=6)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3,3), useMathText=True)
        ax.yaxis.get_offset_text().set_size(5)
        if biflag:
            ax.plot(VClist, R1[:,0], 'o', c='blue',markersize=1)
            ax.plot(VClist, R1[:,1], 'o', c='green',markersize=1)
            ax.errorbar(VClist, R1[:,0], yerr=err[:,0], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            ax.errorbar(VClist, R1[:,1], yerr=err[:,1], fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
        else:
            ax.plot(VClist, R1, 'o', c='blue',markersize=1)
            ax.errorbar(VClist, R1, yerr=err, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
        #x log scale
        ax.set_xscale('log')
        plt.savefig(dir_result+'/R1_'+str(i+1)+'.png', dpi=600)
        plt.close()

        if biflag:
            column_names = "VCLIST,R1a,err1,R1b,err2"
            np.savetxt(
                        f"{dir_result}/R1_"+str(i+1)+".csv",
                        np.column_stack((VClist, R1[:,0], err[:,0], R1[:,1], err[:,1])),
                        delimiter=",",
                        header=column_names,
                        comments=""
                    )
        else:
            column_names = "VCLIST,R1,err"
            np.savetxt(
                        f"{dir_result}/R1_"+str(i+1)+".csv",
                        np.column_stack((VClist, R1, err)),
                        delimiter=",",
                        header=column_names,
                        comments=""
                    )

def fit_Int_fromIR(Int_matrix, delays, dir, Err_matrix=None):

    def IR_residue_err(t, t1, exp, err):
        model = IR_curve(t, t1)
        A = np.sum(model*exp)/np.sum(model**2)
        return (A*model-exp)/err, A*model

    def f_residue(param, delays, Int, Err, idx, result=False):
        par = param.valuesdict()
        res, fmodel = IR_residue_err(delays, 10**par['t1_'+str(idx+1)], Int, Err)
        if result:
            return res, fmodel
        else:
            return res

    x = delays

    param = lmfit.Parameters()
    for i in range(Int_matrix.shape[1]):
        param.add('t1_'+str(i+1), value=-2, min=-5, max=0, vary=False)
        
    for idx in range(Int_matrix.shape[1]):
        param['t1_'+str(idx+1)].set(vary=True)

        Int = Int_matrix[:,idx]
        Err = Err_matrix[:,idx]
        minner = lmfit.Minimizer(f_residue, param, fcn_args=(x, Int, Err, idx))
        result = minner.minimize(method='leastsq', max_nfev=30000)

        popt = result.params
        pcov = result.covar  
        perr = np.sqrt(np.diag(pcov))[0]
        res, fmodel = f_residue(popt, x, Int, Err, idx, result=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, Int, 'o', c='b', label='experimental data')

        err = np.abs((10**(-perr+popt["t1_"+str(idx+1)].value))-10**popt["t1_"+str(idx+1)].value)
        ax.plot(x, fmodel, '--', c='r', label='y = A*(1-2*exp(-t/T1))\nT1 = '+rf'({10**popt["t1_"+str(idx+1)].value:.4e} $\pm$ {err:.3e})'+' s')
          
        err_bar = np.abs(Err)
        ax.errorbar(x, Int, yerr=err_bar,color='k', fmt='none', ecolor='k', elinewidth=0.3, capsize=2, capthick=0.3)

        plt.legend()
        plt.xlabel('Delay (s)')
        plt.ylabel('Intensity (a.u.)')
        plt.savefig(dir+'IR_fit_'+str(idx+1)+'topspin_LNS.png')
        plt.close()

        param['t1_'+str(idx+1)].set(vary=False)

def fit_shiftvsT(shift_tot, temp, dir_result, err_bar=None):

    import scipy.constants

    def calc_shift(temp, S1, S2, A_h, J, gamma=267/4):

        def energy(Si, J):
            return 1/2*J*(Si*(Si+1))

        conv = 1.9865e-23 #lo moltiplico per cm-1 per ottenere J
        kB = scipy.constants.k
        muB = scipy.constants.physical_constants['Bohr magneton'][0]
        ge = 2.0023
        pref = 2*np.pi*ge*muB/(3*kB*gamma*temp)

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

        A = [par['A_'+str(i)] for i in range(len(temp))]
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
            plt.savefig(dir_result+'/shiftvsT1.png')
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
            plt.savefig(dir_result+'/shiftvsT2.png')
            plt.close()
            return conshift
        else:
            return res

    J = 364  #in cm-1
    A_h = [1.127, 0.640, 1.479, 1.1816, 1.278, 1.90, 1.736, 2.2]#0.8  #MHz

    dia_shift = np.array([40.5, 58, 40.5, 58, 58, 40.5, 58, 40.5])

    dia_shift = np.tile(dia_shift, (len(temp),1))
    shift_tot = shift_tot-dia_shift
    print(shift_tot)
    param = lmfit.Parameters()
    param.add('J', value=J, min=0, max=1000, vary=False)
    [param.add('A_'+str(i), value=A_h[i], min=0, max=2, vary=False) for i in range(len(temp))]
    print('shift tot\n',shift_tot)
    minner = lmfit.Minimizer(T_model, param, fcn_args=(temp, shift_tot))
    result = minner.minimize(method='leastsq', max_nfev=30000)
    popt = result.params
    calc_shift = T_model(popt, temp, shift_tot, result=True, T_long=np.arange(200, 20000, 10))


#========================================== ON DEV ===============================================

def fit_int(path, dir_result, list_path, delays_list):

    for i in range(len(delays_list)):
        if 'pdata' not in list_path[i]:
            list_path[i] = list_path[i]+'/pdata/1'

    out_file = [dir_result+'/'+num+'/'+dir_result[:dir_result.index('_modelfit')]+'_sp'+num+'.out' for num in list_path]
    ppm1_list, ppm2_list = read_lim(out_file[0])
    
    t1_intensity = []
    t1_area = []
    
    for i in range(len(out_file)):  
        
        ppm_scale = nmr_spectra_pseudo2d(path+'/'+list_path[i])[1]

        delays = delays_list[i][delays_list[i].argsort()]
        
        sp_files = [dir_result+'/'+list_path[i][:list_path[idx].index('/pdata')]+'/spbsl_D'+str(num+1)+'.csv' for num in range(len(delays))]
        
        t1_intensity.append([])
        t1_area.append([])
        
        for j in range(len(ppm1_list)): 
            
            file_csv = open(dir_result+'/'+list_path[i][:list_path[idx].index('/pdata')]+'/'+str(j+1)+'.csv', 'w')
            
            sx, dx, zero = find_limits(ppm1_list[j], ppm2_list[j], ppm_scale)
            
            intensity = []
            area = []
            for k in range(len(sp_files)): #per ogni delay
                
                spettro = np.loadtxt(sp_files[k])
                spettro_cut = spettro[sx:dx]
                
                intensity.append(np.max(spettro[sx:dx].real))
                AA = np.trapz(spettro[sx:dx])
                area.append(AA)

                file_csv.write(str(delays[k])+'\t'+str(AA)+'\n')

            file_csv.close()
            
            # mono, bi, report1, report2, err1, err2 = fit_exponential(delays, intensity, name =  dir_result+'/'+list_path[i]+'/'+str(j+1)+'_intensitynew')
            # t1_intensity[i].append(10**tuple(mono)[0])

            mono, bi, report1, report2, err1, err2 = fit_exponential(delays, area, name =  dir_result+'/'+list_path[i][:list_path[idx].index('/pdata')]+'/'+str(j+1)+'_areanew')
            t1_area[i].append(10**tuple(mono)[0])
            #print(tuple(bi))
            #print(f'{10**tuple(bi)[1]:.6f}\t{err2[0]}\t{10**tuple(bi)[2]:.6f}\t{err2[1]}\t{tuple(bi)[0]:.6f}')
            
    np.savetxt(dir_result+'/'+'intensity_t1mono.txt', np.array(t1_intensity))
    np.savetxt(dir_result+'/'+'area_t1mono.txt', np.array(t1_area))
    
def model_fit_1D_reg(path, delays_list, list_path, cal_lim = None, dofit=True, prev_fit=None, fast=False, limits1 = None, limits2 = None, L1R = None, L2R = None, doexp=False, f_int_fit=None, fargs=None, reg_args=None):    

    new_dir = os.path.basename(os.path.normpath(path))
    nome_folder = '_modelfitR'   
    try:
        os.mkdir(new_dir+nome_folder)
    except:
        ans = input('\nThe directory '+color_term.BOLD+new_dir+nome_folder+color_term.END+' already exists.\nDo you want to:\n[overwrite the existing directory (1)]\ncreate a new directory (2)\ncontinue writing in the existing directory (3)\n>')
        if ans=='1' or ans == '':
            shutil.rmtree(new_dir+nome_folder)           # Removes all the subdirectories!
            os.makedirs(new_dir+nome_folder)
        elif ans=='2':
            new_dir = input('Write new directory name: ')
            os.makedirs(new_dir+nome_folder)
        elif ans=='3':
            pass
        else:
            print('\nINVALID INPUT')
            exit()
    print('Writing directory '+color_term.BOLD+new_dir+nome_folder+color_term.END)

    dir_res = new_dir+nome_folder
    print('DIR: ', color_term.CYAN+dir_res+color_term.END)

    nameout = new_dir+'.out'

    print('PATH TO SPECTRA: ', path+'/'+dir_res)

    data_d = []
    ppmscale_d = []
    for i in range(len(delays_list)):
        data, ppmscale, ngdicp = nmr_spectra_1d(path+list_path[i])
        data_d.append(data)
        ppmscale_d.append(ppmscale)

    data = np.array(data_d)
    ppm_scale = ppmscale.copy()
    
    acqupars = param_acq(ngdicp)
    procpars = param_pr(ngdicp)  #only qsin and em are considered for the apodization

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
    t_aq = np.linspace(0, td*dw, td) + DE

    to_order = np.hstack((np.reshape(delays_list,(len(delays_list),1)),data))

    to_order = to_order[to_order[:,0].argsort()[::-1]]	

    data = to_order[:,1:]
    delays = to_order[:,0].real

    if cal_lim is not None:
        cal_shift, cal_shift_ppm, data = calibration(ppm_scale, data, cal_lim[0], cal_lim[1]) 
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('CALIBRATION: ('+f'{cal_lim[0]:.5f}'+':'+f'{cal_lim[1]:.5f}'+') ppm\n')
            f.write('in points\n')
            [f.write(str(cal_shift[ii])+'   ') for ii in range(len(cal_shift))]
            f.write('\nin ppm\n')
            [f.write(f'{cal_shift_ppm[ii]:.5f}'+'   ') for ii in range(len(cal_shift_ppm))]
            f.write('\n\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
            f.close()
    else:
        with open(dir_res+'/'+nameout, 'w') as f:
            f.write('SPECTRA PATH: \n')
            f.write(path+'\n')
            f.write('\n')
            f.write('Points:\n')
            [f.write(str(r+1)+'\t'+f'{delays[r]:.3f}'+'\n') for r in range(len(delays))]
            f.close()

    spettro = data[0,:]  #spectra initial guess

    CI = create_input(ppm_scale, spettro)
    filename1 = input1_gen(CI)
    matrix = read_input_INT(filename1)
    shutil.copy2(filename1, dir_res+'/')
    matrix = read_input_INT(filename1)
    multiplicity = matrix[:,-1]

    ##INPUT2 GENERATION##
    filename2 = input2_gen(CI, matrix, acqupars, procpars)
    shutil.copy2(filename2, dir_res+'/')
    tensore = read_input_INT2(filename2, matrix[:,:-1])

    name_peak = np.zeros(tensore.shape[0], dtype='int64')
    name=0
    prev = 0
    for i in range(tensore.shape[0]):
        if multiplicity[i]==0:
            name += 1
            name_peak[i] = name
            prev=0
        elif multiplicity[i]!=0:
            if prev==multiplicity[i]:
                name_peak[i] = name
            else:
                prev = multiplicity[i]
                name += 1
                name_peak[i] = name

    with open(dir_res+'/'+nameout, 'a') as f:
        f.write('\nINPUT1: '+filename1+'\n')
        file = open(filename1).readlines()
        f.write('n. peak\t'+file[0])
        [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
        f.write('\nINPUT2: '+filename2+'\n')
        file = open(filename2).readlines()
        f.write('n. peak\t'+file[0])
        [f.write(str(name_peak[r-1])+'\t'+file[r]) for r in range(1,len(file))]
        [f.write('=') for r in range(100)]
        f.write('\n')
        [f.write('=') for r in range(100)]
        f.write('\n')
        f.write('(I: fit interval, P: N. point)\n')
        f.close()

    ppm1 = matrix[:,1]
    ppm2 = matrix[:,2]
    ppm1_set = list(set(ppm1))
    ppm2_set = list(set(ppm2))
    ppm1_set.sort()
    ppm2_set.sort()

    tensore = np.append(tensore, np.reshape(multiplicity, (len(multiplicity),1)), axis=1)

    tensor_red_list = []

    shift_tot = []
    integral_tot = []
    error_tot = []
    for i in range(len(ppm1_set)):  #per intervallo

        ppm1 = ppm1_set[i]
        ppm2 = ppm2_set[i]
        print('FIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2))

        tensor_red = np.array([tensore[ii,:] for ii in range(tensore.shape[0]) if tensore[ii,1]==ppm1 and tensore[ii,2]==ppm2])
        tensor_red_list.append(tensor_red)

        param = lmfit.Parameters()
        for j in range(tensor_red.shape[0]):

            if tensor_red[j,0]=='true':
                param.add('shift_'+str(j+1), value=tensor_red[j,3], min=tensor_red[j,3]-delta, max=tensor_red[j,3]+delta)
                param.add('k_'+str(j+1), value=tensor_red[j,4])
                param.add('lw_'+str(j+1), value=tensor_red[j,5],  min=1e-4, max = 3.5)
                param.add('ph_'+str(j+1), value=tensor_red[j,6], min=-np.pi, max=np.pi)
                param.add('xg_'+str(j+1), value=0.2, min=0, max=1)
            else:
                param.add('shift_'+str(j+1)+'_f', value=tensor_red[j,3], min=tensor_red[j,3]-delta, max=tensor_red[j,3]+delta)
                param.add('k_'+str(j+1)+'_f', value=tensor_red[j,4])
                param.add('lw_'+str(j+1)+'_f', value=tensor_red[j,5],  min=1e-4, max = 3.5)
                param.add('ph_'+str(j+1)+'_f', value=tensor_red[j,6], min=-np.pi, max=np.pi)
                param.add('xg_'+str(j+1)+'_f', value=0.2, min=0, max=1)

        param.add('A', value=tensor_red[-1,7])
        param.add('B', value=tensor_red[-1,8])
        param.add('C', value=tensor_red[-1,9]) 
        param.add('D', value=tensor_red[-1,10])
        param.add('E', value=tensor_red[-1,11])

        #### read
        if not dofit:
            param = lmfit.Parameters()
            file = open(new_dir+'popt_I'+str(i), 'r')
            param.load(file)
        ###
        
        peak_int, int_err, prev_param_compl, result, _ = fit_peaks_bsl_reg(param, ppm_scale, data, tensor_red, 
                                                        t_aq, sf1, o1p, td, dw, i, dir_res, new_dir, SR=SR, 
                                                        SI=SI, SW=SW, LB=LB, SSB=SSB, dofit=dofit, L1R=L1R, L2R=L2R,
                                                        f_reg=f_reg, reg_args=reg_args)
        
        #### write
        if dofit:
            file = open(dir_res+'/'+new_dir+'popt_I'+str(i), 'w')
            prev_param_compl.dump(file)
            file = open(new_dir+'popt_I'+str(i), 'w')
            prev_param_compl.dump(file)
        ####

        integral_tot.append(peak_int)
        error_tot.append(int_err)
        
        with open(dir_res+'/'+nameout, 'a') as f:
            f.write('\n')
            [f.write('-') for r in range(30)]
            f.write('\nFIT RANGE: ({:.2f}:{:.2f}) ppm'.format(ppm1, ppm2)+'\n')
            f.write('I: '+str(i+1)+'\n')
            [f.write('-') for r in range(30)]
            f.write('\n')
            f.write('\nFit Report: \n')
            f.write(lmfit.fit_report(result)+'\n')
            f.write('\nIntegrals: \n')
            for jj in range(len(peak_int)):
                f.write(str(peak_int[jj])+' +/- '+str(np.abs(int_err[jj]))+'\n')
            f.close()
                
    integral=np.array(integral_tot)
    error = np.array(error_tot) 

    int_del = np.column_stack((integral, delays))  #(n. delays x [integral[:,0],...,integral[:,n], delays[:]])
    order = int_del[:,-1].argsort()
    int_del = int_del[order]
    try:
        error = np.reshape(error[order], (len(error),))
    except:
        error = np.reshape(error[order], (len(error),error.shape[-1]))
        error = error[:,0]
    
    if doexp==True:

        n_peak = 0
        mono_fit = []
        bi_fit = []
        errmono_fit = []
        errbi_fit = []
        fitparameter_f = []
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            mono, bi, report1, report2, err1, err2 = fit_exponential(int_del[:,-1], int_del[:,ii], dir_res+'/'+f'{n_peak}', err_bar=error)
            
            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\nMono\n')
                f.write('y = a + A exp(-t/T1)\nfit: T1=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(mono))
                f.write(report1+'\n')
                f.write('\nBi\n')
                f.write('y = a + A (f exp(-t/T1a)+ (1-f) exp(-t/T1b))\nfit: f=%5.3e, T1a=%5.4e, T1b=%5.4e, a=%5.3e, A=%5.3e\n' % tuple(bi))
                f.write(report2+'\n')
                f.close()

            mono_fit.append(10**tuple(mono)[0])
            errmono_fit.append(err1)
            bi_fit.append((10**tuple(bi)[1], 10**tuple(bi)[2]))
            errbi_fit.append(err2)
            fitparameter_f.append(tuple(bi)[0])
            
            
        with open(dir_res+'/'+'t1.txt', 'w') as f:
            f.write('n.peak\tT1 (s)\terr (s)\tf\n')
            for ii in range(len(mono_fit)):
                try:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+f'{errmono_fit[ii]:.4e}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{mono_fit[ii]:.4e}'+'\t'+'Nan'+'\n')
                try:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+f'{errbi_fit[ii][0]:.4e}'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+f'{errbi_fit[ii][1]:.4e}'+'\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
                except:
                    f.write(str(ii+1)+'\t'+f'{bi_fit[ii][0]:.4e}'+'\t'+'Nan'+'\t'+f'{bi_fit[ii][1]:.4e}'+'\t'+'Nan\t'+f'{fitparameter_f[ii]:.4f}'+'\n')
            f.close()
        with open(dir_res+'/'+nameout, 'a') as f:
            for r in range(100):
                f.write('=')
            f.close()

    elif f_int_fit is not None and doexp==False:

        n_peak = 0
        for ii in range(int_del.shape[-1]-1):
            n_peak += 1
            #f_int_fit must return a dictionary with the fit parameters and the model
            parameters, model = f_int_fit(int_del[:,-1], int_del[:,ii], **fargs, err_bar=error)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            ax.plot(int_del[:,-1], model, 'r--', lw=0.7)
            
            #ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
            plt.close()

            with open(dir_res+'/'+nameout, 'a') as f:
                f.write('\n')
                [f.write('-') for r in range(30)]
                f.write('\nN. PEAK: '+f'{n_peak}'+'\n')
                [f.write('-') for r in range(30)]
                f.write('\n')
                f.write('\nFit Parameters:\n')
                f.write('\n')
                f.write(' '.join([f'{key}={val}' for key, val in parameters.items()])+'\n')
                f.close()

    else:

        for ii in range(int_del.shape[-1]-1):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(int_del[:,-1], int_del[:,ii], yerr=error, fmt='none', ecolor='k', elinewidth=0.2, capsize=2, capthick=0.2)
            ax.plot(int_del[:,-1], int_del[:,ii], 'o', c='blue')
            #ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2), useMathText=True)
            plt.savefig(dir_res+'/Peak_'+str(ii+1)+'.png', dpi=600)
            plt.close()


    return dir_res

def fit_peaks_bsl_reg(param, ppm_scale, spettro0, tensor_red, t_aq, sf1, o1p, td, dw, j, jj, dir_res, new_dir, SR=0, SI=0, SW=0, LB=0, SSB=0, dofit=True, IR=False, L1R=None, L2R=None, f_reg=None, reg_args=None):

    cal = np.abs(ppm_scale[0]-ppm_scale[1])+SR/sf1

    cycle = -1
    def f_residue(param, ppm_scale, data, tensor_red_list, delays, result=False):
        nonlocal cycle
        cycle += 1
        print('fit peaks: '+f'{cycle:5g}',end='\r')

        par = param.valuesdict()

        integral_list_global = []
        err_list_global = []
        residuals_global = []

        for idx in range(data.shape[0]):  #per delay

            lor_list = []
            lor_ph0_list = []
            comp_list = []
            prev=0
            spettro = data[idx,:].copy()
            sim_spectra_tot = np.zeros_like(spettro, dtype='float64')

            res_tot = []
            res_tot_gradient = []

            for ii in range(len(tensor_red_list)):  #per ogni intervallo
                
                tensor_red = tensor_red_list[ii]
                mult = tensor_red[:,-1]
                
                lor_list.append([])
                lor_ph0_list.append([])
                comp_list.append([])

                sx, dx, _ = f_fun.find_limits(tensor_red[0,1], tensor_red[0,2], ppm_scale)

                sim_spectra = np.zeros_like(spettro, dtype='float64')
                
                for jj in range(tensor_red.shape[0]):  #per ogni voigt

                    lor = f_fun.t_voigt(t_aq, (par['shift_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)]-cal-o1p)*sf1, 2*np.pi*par['lw_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)]*sf1,
                                                    A=par['k_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)], phi=par['ph_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)], x_g=par['xg_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)])

                    ### processing
                    lor *= f_fun.em(lor, LB, SW)
                    lor *= f_fun.qsin(lor, SSB)
                    lor = f_fun.zf(lor, SI)
                    ###

                    #mi serve per l'errore 
                    lor_ph0 = f_fun.t_voigt(t_aq, (par['shift_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)]-cal-o1p)*sf1, 2*np.pi*par['lw_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)]*sf1,
                                                A=np.abs(par['k_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)]), phi=0, x_g=par['xg_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)])
                    ### processing
                    lor_ph0 = f_fun.zf(lor_ph0, SI)
                    ###

                    lor = f_fun.ft(lor, SI, dw, o1p, sf1)[0]
                    lor = np.conj(lor)[::-1].real

                    sim_spectra += lor
                    comp_list[ii].append(lor[sx:dx])

                    lor_ph0 = f_fun.ft(lor_ph0, SI, dw, o1p, sf1)[0]
                    lor_ph0 = np.conj(lor_ph0[::-1]).real

                    m=mult[jj]
                    if m==0:
                        prev=0
                        lor_list[ii].append(lor)
                        lor_ph0_list[ii].append(lor_ph0)
                    elif m!=0:
                        if m==prev:
                            lor_m = lor+lor_list[ii][-1]
                            lor_m_int = lor_ph0+lor_ph0_list[ii][-1]
                            del lor_list[ii][-1]
                            lor_list[ii].append(lor_m)
                            del lor_ph0_list[ii][-1]
                            lor_ph0_list[ii].append(lor_m_int)
                        else:
                            prev = m
                            lor_list[ii].append(lor)
                            lor_ph0_list[ii].append(lor_ph0_list)

                sim_spectra_tot += sim_spectra

                x = ppm_scale[sx:dx]
                corr_baseline = par['E_'+str(idx)]*x**4 + par['D_'+str(idx)]*x**3 + par['C_'+str(idx)]*x**2 + par['B_'+str(idx)]*x + par['A_'+str(idx)]

                if ii==0:
                    #cost deve essere per forza unico, altrimenti non posso definire la baseline totale
                    cost = np.sum((sim_spectra[sx:dx]+corr_baseline)*spettro[sx:dx].real)/np.sum((sim_spectra[sx:dx]+corr_baseline)**2)

                lor_ph0_list[ii] = np.array(lor_ph0_list[ii])*np.abs(cost)  
                lor_list[ii] = np.array(lor_list[ii])*cost
                comp_list[ii] = np.array(comp_list[ii])*cost

                model = cost*(corr_baseline+sim_spectra[sx:dx]) 

                res = model-spettro[sx:dx]

                res_tot.append(res/max(spettro[sx:dx]))
                res_tot_gradient.append((np.gradient(model)-np.gradient(spettro[sx:dx]))/max(np.gradient(spettro[sx:dx])))

                if cycle%1000==0 or result:
                    # f_fun.f_figure_comp(ppm_scale[sx:dx], spettro[sx:dx], model, comp_list[ii], 
                                        # name=dir_res+'/'+new_dir+'_D'+str(idx+1)+'_I'+str(ii+1)+'_2ndround', basefig = cost*corr_baseline, 
                                        # dic_fig={'h':5.59,'w':4.56, 'sx':tensor_red[0,1], 'dx':tensor_red[0,2]})
                    f_fun.f_figure_comp(ppm_scale[sx:dx], spettro[sx:dx], model, comp_list[ii], 
                                    name=dir_res+'/'+new_dir+'_D'+str(idx+1)+'_I'+str(ii+1)+'_2ndround', basefig = cost*corr_baseline, 
                                    dic_fig={'h':5.59,'w':4.56, 'sx':tensor_red[0,1], 'dx':tensor_red[0,2]})
                    f_fun.f_figure_comp(ppm_scale[sx:dx], np.gradient(spettro[sx:dx]), np.gradient(model), [np.gradient(comp_list[ii][qq]) for qq in range(len(comp_list[ii]))], 
                                    name=dir_res+'/'+new_dir+'_D'+str(idx+1)+'_I'+str(ii+1)+'_deriv'+'_2ndround', basefig = cost*np.gradient(corr_baseline), 
                                    dic_fig={'h':5.59,'w':4.56, 'sx':tensor_red[0,1], 'dx':tensor_red[0,2]})
            
            integral_in=[]
            int_err = []
            for ii in range(len(tensor_red_list)):  #per intervallo
                integral_in.append([])
                int_err.append([])
                tensor_red = tensor_red_list[ii]
                sx, dx, _ = f_fun.find_limits(tensor_red[0,1], tensor_red[0,2], ppm_scale)
                for jj in range(len(lor_list[ii])):  #per voigt
                    integral_in[ii].append(np.trapz(lor_list[ii][jj]))
                    Err = error_calc_num(ppm_scale, res_tot[ii]*max(spettro[sx:dx]), lor_ph0_list[ii][jj], np.trapz(lor_ph0_list[ii][jj]), par['shift_I'+str(ii+1)+'_V'+str(jj+1)+'_'+str(idx)], sx, dx)
                    int_err[ii].append(Err)
                    

            integral_list_global.append(integral_in)  #delay, interval, peak
            err_list_global.append(int_err)

            residuals = []
            for idxr,r in enumerate(res_tot):
                residuals = np.concatenate([residuals, r])
                residuals = np.concatenate([residuals, res_tot_gradient[idxr]])

            x = ppm_scale
            corr_baseline = par['E_'+str(idx)]*x**4 + par['D_'+str(idx)]*x**3 + par['C_'+str(idx)]*x**2 + par['B_'+str(idx)]*x + par['A_'+str(idx)]
            model = cost*(corr_baseline+sim_spectra_tot)
            deriv = (np.gradient(corr_baseline) + np.gradient(np.gradient(corr_baseline))*1e2 + np.gradient(np.gradient(np.gradient(corr_baseline)))*1e4)
                
            if cycle%1000==0 or result:
                f_fun.f_figure(ppm_scale, spettro, model, name=dir_res+'/'+new_dir+'_D'+str(idx+1)+'_2ndround', basefig = cost*corr_baseline)
                
            
            # if bsl_range is not None:
            #     bsl_sect = []
            #     for lims in bsl_range:
            #         sx, dx, _ = f_fun.find_limits(lims[0], lims[1], ppm_scale)
            #         x = ppm_scale[sx:dx]
            #         corr_baseline = par['E_'+str(idx)]*x**4 + par['D_'+str(idx)]*x**3 + par['C_'+str(idx)]*x**2 + par['B_'+str(idx)]*x + par['A_'+str(idx)]
            #         bsl_sect.append(corr_baseline*cost-spettro[sx:dx])
            #     bsl_sect = np.concatenate(bsl_sect)
            #     residuals = np.concatenate((residuals/max(spettro)))#, deriv, bsl_sect*cost/max(spettro)/10))  #non ci ho messo la regolarizzazione con derivata perché ho già la regolarizzazione con il t1
            # else:
            #     residuals = np.concatenate((residuals/max(spettro)))#, deriv))
            # residuals_global.append(residuals)
        
            # residuals = np.concatenate((residuals, deriv))
            residuals_global.append(residuals)
        
        integral_list_global = np.array([item for sublist in integral_list_global for item in sublist])
        err_list_global = np.array([item for sublist in err_list_global for item in sublist])

        #### inversion recovery fit    
        R1_list = []
        res_tot = []
        fmodel_list = []
        for idx in range(integral_list_global.shape[1]):  #per ogni picco
            res, fmodel = IR_residue(delays, 10**par['t1_'+str(idx+1)], integral_list_global[:,idx])
            res_tot.append(res*10)
            R1_list.append(1/10**par['t1_'+str(idx+1)])
            fmodel_list.append(fmodel)

        fmodel_list = np.array(fmodel_list)

        res_IR = np.concatenate(res_tot)
        residuals_global.append(res_IR)

        if cycle%1000==0 or result:

            print(R1_list)

            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.plot(np.concatenate(residuals_global))
            ax = fig.add_subplot(122)
            ax.plot(res_IR)
            plt.savefig('res_global.png')
            plt.close()

            f_fun.f_figure_decays(delays, integral_list_global, exp_model = fmodel_list, R1_list=R1_list, name=dir_res+'/'+new_dir+'_IR', logscale=False)
            
        if not result:
            return np.concatenate(residuals_global)
        else:
            return integral_list_global, err_list_global, R1_list
           
    minner = lmfit.Minimizer(f_residue, param, fcn_args=(ppm_scale, data, tensor_red_list, delays))
    # result = minner.minimize(method='differential_evolution', max_nfev=30000)
    result = minner.minimize(method='Nelder', max_nfev=10000)#, xtol=1e-7, ftol=1e-7)
    params = result.params
    result = minner.minimize(params=params, method='leastsq', max_nfev=30000)#, xtol=1e-7, ftol=1e-7)
    
    popt = result.params
    Int, Err, R1 = f_residue(popt, ppm_scale, data, tensor_red_list, delays, result=True)
    
    return Int, Err, R1
