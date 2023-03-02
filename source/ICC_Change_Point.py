"""
Introduction to Change-Point ICC Code


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.weightstats import ztest
from  ICC_Find import *
from ICC_Curve import *
import scipy.fft as FFT
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm


class Change_Point_Analysis:
    """
    Class for performing change-point analysis on epidemic case data.
    
    Attributes
    ----------
    case : ndarray
        case data
    inc : ndarray
        incidence data
    inc_smooth : ndarray
        smoothed incidence data
    case_smooth : ndarray
        smoothed case data (found through cumsum(inc_smooth))
    case_interp : ndarray
        interpolated grid of case data
    inc_interp : ndarray
        interpolated incidence data
    avg_size : int
        even integer that is used in smoothing the data, first pass of change point search, and others. Can be adjusted as desired. 
        Should be set at the average recovery time.
    x_win_param_list : list
        list of icc parameters generated from the full-window optimal icc curve in the first pass
    x_win_change_list : list
        list of change point statistics generated in first pass
    change_point_index_list : ndarray
        index values that meet change-point criteria
    change_point_list : list
        list of change points and change intervals (if multiple consecutive change points)
    ICC_curve_list : list
        list of dictionaries for each partition region between change intervals
    
    
    
    Methods
    -------
    smooth_data(smooth_param, smooth_method)
       smooths icc case data in time domain
       
    interp_data(case_frequency, kind='linear')
        interpolates icc data from time domain to case domain
        
    first_pass(window_size, starting_values, Recovery_time, rand_search_count = 100, threshold = 0.01, progress_bar = True)
        performs initial search for change point
        
    find_change_points(threshold, min_change_distance)
        uses results of first-pass to partition case data into intervals via change points
    
    refine_search    
    
    
    
    
    """
    
    def __init__(self,case):
        """
        Input original case and incidence data
        
        Parameters
        ----------
        case : 1darray
            cumulative case data
        inc : 1darray
            incidence (new cases) data
        """
        self.case = case
    
    def find_inc(self, time, day_count, smooth = False, avg_size = None):
        """
        Parameters
        ----------
        time : ndarray
            Array of day values for given cumulative case counts.
            
        day_count : ndarray
            Sampling frequency to calculate incidence. Note refining case is required.
        
        smooth : bool, default False
            Using moving average smoothing method. Requires argument avg_size.
            
        avg_size : int
            Moving window size.     
        """        
        
        if smooth:
            self.avg_size = avg_size
            self.case_smooth = np.empty(len(self.case))
            for index in range(1,len(self.case)):
                window_size = np.amin([index, self.avg_size])
                self.case_smooth[index] = np.sum(self.case[index-int(self.avg_size/2):index+int(self.avg_size/2)])/window_size

            T = time
            Cum_Cases = self.case_smooth
            k = 0
            last = 0
            inc_day = []
            case_count = []
            day = 0
            Rec_count = 0
            Rec_count_last = 0

            for ii in range(2,int(T[-1]/day_count)+1):
                count = 0
                while T[k] <= day+day_count:
                    if k == len(T)-2:
                        break
                    k += 1
                    count += 1
                    if Cum_Cases[k] == Cum_Cases[k-1]:
                        Rec_count += 1
                if k == len(T)-1:
                    break
                if T[k] == T[last]:
                    inc_day.append(0)
                else:   
                    inc_day.append((Cum_Cases[k]-Cum_Cases[last])/(T[k]-T[last]))
                    case_count.append((Cum_Cases[k]+Cum_Cases[last])/2)
                    last += count
                    Rec_count_last = Rec_count
                day += day_count
            self.inc_smooth = np.array(inc_day[:-1])[:-int(avg_size/2)]
            self.case_smooth = np.array(case_count)[:-int(avg_size/2)]
            

        T = time
        Cum_Cases = self.case
        k = 0
        last = 0
        inc_day = []
        case_count = []
        day = 0
        Rec_count = 0
        Rec_count_last = 0

        for ii in range(2,int(T[-1]/day_count)+1):
            count = 0
            while T[k] <= day+day_count:
                if k == len(T)-1:
                    break
                k += 1
                count += 1
                if Cum_Cases[k] == Cum_Cases[k-1]:
                    Rec_count += 1
            if k == len(T)-1:
                break
            if T[k] == T[last]:
                inc_day.append(0)
            else:   
                inc_day.append((Cum_Cases[k]-Cum_Cases[last])/(T[k]-T[last]))
                case_count.append((Cum_Cases[k]+Cum_Cases[last])/2)
                last += count
                Rec_count_last = Rec_count
            day += day_count
        self.inc = inc_day
        self.case = case_count        
        return
    
    def fourier_smooth_data(self, smooth_param):
        """
        Uses spectral smoothing method to smooth incidence data
        
        Parameters
        ----------
        smooth_param : int
            Number of Fourier modes
        """
        num_modes = smooth_param
        Inc_Fourier = FFT.fft(np.concatenate([self.inc, self.inc[::-1]]))
        Inc_Fourier[num_modes:-(num_modes+1)] *= 0
        self.inc_smooth_f = np.real(FFT.ifft(Inc_Fourier))[:len(self.inc)]      
        self.case_smooth_f = np.cumsum(self.inc_smooth_f)

        return
    
    def interpolate_data(self, case_frequency, kind = 'linear', fourier = True):
        """
        Interpolates incidence data to have equidistant case sampling points.
        
        Parameters
        ----------
        case_frequency : int
            The frequency at which cases are interpolated. ie. 1000 -> incidence value for every 1000 cases.
        kind : str, default 'linear'
            Type of interpolation used. Corresponds to 'kind' argument for scipy.interpolate.interp1d.
        fourier : bool, default True
            Use Fourier smooth data or moving-avg smoothing data.
        """
        if fourier:
            inc = self.inc_smooth_f
            case = self.case_smooth_f
        else:
            inc = self.inc_smooth
            case = self.case_smooth
        if case[-1] <= 2000:
            print('Need more than 2000 cases')
            return
        interp_fn = interp1d(case, inc, kind = kind)
        self.case_interp_prepass = np.arange(case[0], case[-1], case_frequency)
        self.inc_interp_prepass = interp_fn(self.case_interp_prepass)
        try:
            no_fourier_threshold = np.where(inc[:100] < 0)[-1][0] + 1
        except:
            no_fourier_threshold = 0 #np.where(case > 2000)[0][0]
        self.beginning_curve_index = np.where(case > self.case_interp_prepass[self.avg_size])[0][0]
        self.case_interp = np.concatenate((case[no_fourier_threshold:self.beginning_curve_index], self.case_interp_prepass[self.avg_size:]))
        self.inc_interp = np.concatenate((inc[no_fourier_threshold:self.beginning_curve_index], self.inc_interp_prepass[self.avg_size:]))      
        return
    
    def first_pass(self, window_size, p_level = 0.01, min_change_distance = 30, threshold = None, combine = True, print_fig = True, fig_size = (8,6)): 
        """
        Performs first pass on dataset to find local ICC fits on a moving window and calculates the change-point statistic at every value.
        
        Notes
        -----
        
        
        
        Paramters
        ---------
        window_size : int
            Size of window for which calculation of change-point. Recommended use 2/gamma or 2*avg recovery time.
        p_level : float, default 0.01
            Significance level for change-point detection
        min_change_distance : int, default 30
            Minimum time between change-points. Recommended to be 1/gamma
        threshold : int, optional
            Maximum vertical distance between neighboring peaks
        print_fig : bool, default True
            Visualize the ICC curve and detected change-points
        
        Return
        ------
        change_points : list
            List of indices where change-points occur.    
        """
        case = self.case_interp
        inc = self.inc_interp
        self.x_normal_cp = np.zeros([len(case),2])
        self.mean_list = np.zeros([len(case)])
        for index, _ in enumerate(inc):
            middle_index = index - int(window_size/2)
            if index >= window_size + 1:
                first_half = inc[index - window_size : middle_index] - inc[index - window_size - 1 : middle_index - 1]
                second_half = inc[middle_index : index] - inc[middle_index - 1 : index - 1]
                z_stat = ztest(first_half, second_half)
                self.x_normal_cp[middle_index-1] = z_stat
                self.mean_list[middle_index-1] = np.mean(second_half) - np.mean(first_half)
                    
        stat = np.array(self.x_normal_cp[:,1])
        change_point = find_peaks(1-self.x_normal_cp[:,1] ,height = 1-p_level, distance = min_change_distance, threshold = threshold)[0]
        
        if combine == True:
            change_points = []
            for index, cp in enumerate(change_point[1:]):
                tracker = 1
                if np.abs(inc[change_point[index]] - inc[cp])/inc[cp] < 0.1:
                    tracker += 1
                else:
                     change_points.append(int(np.mean(change_point[index - tracker + 1 : index + 1])))
            change_point = change_points
        self.change_points = []
        for cp in change_point:
            self.change_points.append(np.argmin(np.abs(case[cp] - self.case_smooth_f)))
            
        if print_fig == True:
            fig, ax1 = plt.subplots(figsize = fig_size)
            ax = ax1.twinx()
            ax1.plot(case,inc, color = 'k', alpha = 1)
            l = 'Potential Change Point'
            for cp in change_point:
                ax.plot([case[cp] ,case[cp]], [0, 1], 'r--', alpha = 0.8, lw = 2, label = l)
                l = ''
            ax.plot(case, self.x_normal_cp[:,1], 'g--', lw = 1, label = 'P-Value')#/(CP.inc_interp))
            ax.set_ylim(0,1.5*p_level)
            ax.set_ylabel('P Value')
            ax1.set_ylabel('Incidence')
            ax.set_xlabel('Cumulative Cases')
            fig.legend(ncol = 2)
            plt.show()
        return self.change_points

    def threshold(self, threshold):
        """
        Splits change-points between those at the end of an ICC curve and those that are mid-curve.
        
        Parameters
        ----------
        threshold : float
            Categorizes change points above and below an incidence threshold.
        
        Returns
        -------
        self.change_point_threshold : list
            List of change points below the threshold.
        """
        case = self.case_smooth_f
        inc = self.inc_smooth_f
        self.change_points_threshold = [cp for cp in self.change_points if inc[cp] < threshold]
        return self.change_points_threshold
    
  
    def find_curves(self, title):
        """
        Identifies ICC Curves and initializes the dictionaries for the fit_curves method.
        
        Parameters
        ----------
        title : str
            The title that is placed as the value for the 'title' key for every ICC_curve dictionary. This can be used to sort/organize dictionaries as desired. 
        Returns
        -------
        ICC_curve_list : list
            List of dictionaries of ICC Curves including start_index, stop_index, shift, and ICC_Find intialization parameters.
        """
        self.ICC_curve_list = []
        try:
            self.change_point_list = self.change_points
        except:
            self.change_point_list = []
        try:
            self.change_point_list_t = self.change_points_threshold
        except:
            self.change_point_list_t = []
        
        
        ICC_curve = {}
        ICC_curve['title'] = title
        try:
            ICC_curve['start_index'] = np.where(self.case_smooth_f < 0)[0][-1] + 1
        except:
            ICC_curve['start_index'] = 1
        shift =  self.case_smooth_f[ICC_curve['start_index']]
        ICC_curve['shift'] = shift 
        ICC_curve['change_points'] = []
        
        for index, cp in enumerate(self.change_point_list):
            if cp in self.change_point_list_t:
                if ICC_curve['start_index'] < cp:
                    ICC_curve['stop_index'] = cp
                    c_inf_center = self.case_smooth_f[ICC_curve['stop_index']]
                    c0_center = 0 #self.case_smooth_f[ICC_curve['start_index']]
                    start_values =  [[0.95*c_inf_center - shift, 3*c_inf_center - shift], [-shift,0.05*shift]]
                    ICC_curve['init_values'] = [self.case_smooth_f[cp], c0_center]
                    ICC_curve['start_values'] = start_values
                    ICC_curve['change_points_labels'] = []

                    self.ICC_curve_list.append(ICC_curve)

                    ICC_curve = {}
                    ICC_curve['title'] = title
                    ICC_curve['start_index'] = cp
                    shift =  self.case_smooth_f[ICC_curve['start_index']]
                    ICC_curve['shift'] = shift
                    ICC_curve['change_points'] = []
                    
            else:
                ICC_curve['change_points'].append(cp)        
                                             
        ICC_curve['stop_index'] = len(self.case_smooth_f) - 1
        shift = self.case_smooth_f[ICC_curve['start_index']] 
        c_inf_center = self.case_smooth_f[ICC_curve['stop_index']]
        c0_center = self.case_smooth_f[ICC_curve['start_index']]
        start_values =  [[0.95*c_inf_center - shift, 2*c_inf_center - shift], [-shift,0.05*shift]]
        ICC_curve['start_values'] = start_values
        ICC_curve['init_values'] = [c_inf_center, ICC_curve['start_index']] - shift
        ICC_curve['change_points_labels'] = []
        
        self.ICC_curve_list.append(ICC_curve)
        
        return self.ICC_curve_list


    def delete_low_curves(self, threshold):
        """
        Eliminates ICC curves that do not go above a prescribed threshold.
        
        Parameters
        ----------
        threshold : float
            Number for which if a the incidence of an ICC curve does not go above this value, the curve is deleted.
            
        Returns
        -------
        ICC_curve_list : list
            List of dictionaries of ICC Curves including start_index, stop_index, shift, and ICC_Find intialization parameters.
        """
        keep_list = []
        for index, curve in enumerate(self.ICC_curve_list):
            start_index = curve['start_index']
            stop_index = curve['stop_index']
            
            if np.amax(self.inc_smooth[start_index : stop_index]) > threshold or self.case_smooth[start_index] < 10*threshold:
                keep_list.append(curve)
        self.ICC_curve_list = keep_list
        return self.ICC_curve_list

    def refined_search(self, ICC_curve_list = None, window_size = None, recovery_time = None, parallel = False):
        """
        Performs a refined search with shifting values at the beginning of each wave.
        We shift whenever self.change_point_list reports a "change interval" rather than a change point.
        -test if shift is better or not
        -re-run windowing for intermediate curves
        
        Parameters
        ----------
        ICC_curve_list : list, default (last list generated in object)
            List of ICC curve dictionaries as generated by "first pass".
            
        window_size : list, default self.avg_size
            Size of sampling window used when performing progressive parameter inference
        
        recovery_time : list, default self.Recovery_time
            Recovery time used for each ICC curve
        
        parallel : bool, default False
            Perform progressive inference in parallel (each ICC curve individually, used joblib package)
        """
        
        if not ICC_curve_list:
            ICC_curve_list = self.ICC_curve_list
        if not window_size:
            window_size = [self.avg_size]
        if not recovery_time:
            recovery_time = [self.avg_size]
        if len(window_size) == 1:
            window_size_list = np.ones(len(ICC_curve_list))*window_size[0]
        if len(recovery_time) == 1:
            recovery_time_list = np.ones(len(ICC_curve_list))*recovery_time[0]
        
        if len(window_size) != len(ICC_curve_list):
            print('Window Size Argument is not the right shape. Should be 1 or length of curve list')
        if len(recovery_time) != len(ICC_curve_list):
            print('Recovery Time Argument is not the right shape. Should be 1 or length of curve list')
        
        if parallel == False:
            self.x_win_param_list = []
            self.x_win_loss_list = []
            self.x_win_change_list = []
            for curve_index, curve in enumerate(ICC_curve_list):
                recovery_time = recovery_time_list[curve_index]
                window_size = window_size_list[curve_index]
                params, loss, change = self.window_inference(curve_index, curve, recovery_time, window_size)
                self.x_win_param_list.append(params)
                self.x_win_loss_list.append(loss)
                self.x_win_change_list.append(change)
        else:
            results = Parallel(n_jobs=ncores)(delayed(self.window_inference)(curve_index, curve, recovery_time_list[curve_index], window_size_list[curve_index]) for curve_index, curve in enumerate(ICC_curve_list))    
            results = np.row_stack(results)
            self.x_win_param_list = list(results[:,0])
            self.x_win_loss_list = list(results[:,1])
            self.x_win_chnage_list = list(results[:,2])
        return self.x_win_change_list 
    
    def fit_curves(self, ICC_curve_list = None, recovery_time_list = None, split_by_change_point = False, overwrite = True):
        """
        Fits ICC curves from given list (or last list generated in the object)
        
        Parameters
        ----------
        ICC_curve_list : list, default (last list generated in object)
            List of ICC curve dictionaries as generated by "first pass".
            
        recovery_time_list : list, default self.Recovery_time
            Either single element list or one the length of the ICC curves if multiple recovery times is desired.
            
        overwrite : bool, default True
            Overwrites self.ICC_curve_list with the results.
            
        Returns
        -------
        ICC_curve_list : list
            List of ICC curve dictionaries.
        """
        
        if ICC_curve_list == None:
            ICC_curve_list = self.ICC_curve_list
        if recovery_time_list == None:
            recovery_time_list = self.Recovery_time*np.ones(len(ICC_curve_list))
        else:
            self.Recovery_time = recovery_time_list[0]
        if len(recovery_time_list) == 1:
            recovery_time_list = recovery_time_list[0]*np.ones(len(ICC_curve_list))
        elif len(recovery_time_list) != len(ICC_curve_list):
            print('WARNING: Recovery time list needs to be length of ICC_curve_list or 1')
            return 
                
        param_list = []
        for index, curve in enumerate(ICC_curve_list):
            # curve = ICC_curve_list[index]
            start_index = curve['start_index']
            stop_index = curve['stop_index']
            shift = curve['shift']
            recovery_time_list = [12]*np.ones(len(ICC_curve_list))
            ICC_curve_list[index]['curve_obj_cp'] = []
            
            if split_by_change_point == True:
                try:
                    change_points_labels = curve['change_points_labels']
                except:
                    print('WARNING: You must first run the change_point_check method')
                    return
                for cp_index, cp in enumerate(curve['change_points']):
                    if change_points_labels[cp_index] == 'Change Point' and cp - start_index > self.avg_size:
                        Inc = self.inc_smooth_f[start_index:cp]
                        case = self.case_smooth_f[start_index:cp] - shift
                        start_values = [[0.95*case[-1], 3*case[-1]], [case[0]-shift,case[0]+0.05*shift]]
                        init_values = [np.mean([0.95*case[-1], 3*case[-1]]), np.mean([case[0]-shift,case[0]+0.05*shift])]


                        test_infer = ICC_Find(case, Inc, recovery_time_list[index], start_values, init_values = init_values, parallel = False, rand_search_count = 1000)

                        test_infer_list = np.array(test_infer)
                        arg_min = np.argmin(test_infer_list[:,1])
                        test_infer = test_infer_list[arg_min, 0]

                        ICC_curve_list[index]['curve_obj_cp'].append(ICC_Curve(test_infer))

                        start_index = cp

            Inc = self.inc_smooth_f[start_index:stop_index]
            case = self.case_smooth_f[start_index:stop_index] - shift
            if split_by_change_point == True:
                
                # init_values = [case[-1], case[0]]
                init = [0,0]
                try:
                    p = curve_fit(icc, case, Inc)[0]
                    i=1
                    c0_range = np.linspace(-i*shift, 1.01*case[0])
                    while icc(c0_range[0], *p)>0:
                        i+=1
                        c0_range = np.linspace(-i*shift, case[0])
                        if i == 5:
                            break
                    try:        
                        init[1] = c0_range[icc(c0_range, *p) < 0][-1]
                    except:
                        init[1] = np.mean(c0_range)
                    i=1
                    c_inf_range = np.linspace(case[-1], i*case[-1])
                    while icc(c_inf_range[0], *p)>0:
                        i+=1
                        c_inf_range = np.linspace(case[-1], i*case[-1])
                        if i == 3:
                            break
                    try:
                        init[0] = c_inf_range[icc(c_inf_range, *p) < 0][0]
                    except:
                        init[0] = np.mean(c_inf_range)
                    start_values = [[c_inf_range[0], c_inf_range[-1]], [c0_range[0], c0_range[1]]]
                except:
                    init = [case[0],case[1]]
                    start_values = curve['start_values']
            else:
                start_values = curve['start_values']
                init = curve['init_values']
            test_infer = ICC_Find(case, Inc, recovery_time_list[index], start_values, init_values = init, parallel = False, rand_search_count = 500)
            test_infer_list = np.array(test_infer)
            arg_min = np.argmin(test_infer_list[:,1])
            test_infer = test_infer_list[arg_min, 0]
            ICC_curve_list[index]['curve_obj_cp'].append(ICC_Curve(test_infer))        
        if overwrite == True:
            self.ICC_curve_list = ICC_curve_list
        return ICC_curve_list
    
    def check_change_points(self, ICC_curve_list = None, recovery_time_list = None, print_fig = True, parallel = False, ncores = 2, overwrite = True, fname = None):
        """
        Determines if a point of significant change in incidence is either a peak or a change point
        
        Parameters
        ----------
        ICC_curve_list : list, optional
            List of dictionaries of ICC curves. If none is given, most recent ICC_curve_list is used.
            
        recovery_time_list : list, optional
            List of integers. If none is given, initial recovery time is used. 
            
        Yields
        ------
        curve['change_point_labels'] : list
            List of 'Peak' or 'Change Point' for each change point in curve['change_points'].
            
        """
        if ICC_curve_list == None:
            ICC_curve_list = self.ICC_curve_list
        if recovery_time_list == None:
            recovery_time_list = self.Recovery_time*np.ones(len(ICC_curve_list))
        elif len(recovery_time_list) == 1:
            recovery_time_list = recovery_time_list[0]*np.ones(len(ICC_curve_list))
            self.Recovery_time = recovery_time_list[0]
        self.cp_smooth_list = []
        if parallel == False:    
            for curve_index, curve in enumerate(ICC_curve_list):
                if curve['change_points'] != []:
                    change_points_labels = []
                    if print_fig == True:
                        fig, ax1 = plt.subplots()
                        ax1.plot(self.case_smooth_f[curve['start_index'] : curve['stop_index']], self.inc_smooth_f[curve['start_index'] : curve['stop_index']], 'k',lw = 1, label = 'Data')
                        ax = ax1.twinx()
                        ax1.set_ylabel('Incidence')
                    for cp_index, change_point in enumerate(curve['change_points']):
                        try: 
                            if change_point - 3*self.avg_size > curve['change_points'][cp_index - 1]:
                                init = last_param[2:]
                                init[0] = np.amax([1,init[0]])
                            else:
                                
                                init = None
                        except:
                            init = None
                        case_interp, inc_interp, change_list, last_param = self.window_inference(curve_index, curve, recovery_time_list[curve_index], self.avg_size, peak_or_change = True, change_point = change_point, init = init)
                        smooth_size = self.avg_size/2
                        cl_smooth = np.empty(len(change_list))
                        for index in range(1,len(change_list)):
                            window_size = np.amin([index, smooth_size])
                            cl_smooth[index] = np.sum(change_list[index-int(self.avg_size/2):index+int(self.avg_size/2)])/window_size
                        
                        cl_smooth = cl_smooth[6:-6]
                        case_interp = case_interp[6:-6]
                        change_index = np.argmin(np.array(cl_smooth))
                        new_index = np.argmin(np.abs(self.case_smooth_f - case_interp[change_index]))
                        check_window = [np.amax([0,len(cl_smooth)/2 - 1*self.avg_size]), len(cl_smooth)/2 + 1*self.avg_size]
                        first = cl_smooth[:self.avg_size]
                        last = cl_smooth[-self.avg_size:]
                        mid = cl_smooth[int(check_window[0]):int(check_window[1])]
                        first_err = np.mean(first)
                        last_err = np.mean(last)
                        side = np.append(first, last)
                        var = np.var(side)
                        self.cp_smooth_list.append(cl_smooth)
                        # print(check_window, change_index, np.amin(cl_smooth), var, first_err, last_err, np.argmax(inc_interp))
                        switch = False
                        skip = False
                        if np.abs(first_err - last_err) < 1:
                            if np.amin(mid) < np.mean([first_err, last_err])-3:#  np.amin(cl_smooth) > 5*np.sqrt(var):
                                switch = True
                                print('variance')
                            else:
                                skip = True
                        elif change_index > check_window[0] and change_index < check_window[-1] and not skip:
                            switch = True
                            print('min')
                        if switch == False and np.argmax(inc_interp) > check_window[0] and np.argmax(inc_interp) < check_window[-1] and check_window[0] != 0:
                            change_points_labels.append('Peak')
                            c = 'purple'
                            l = 'Peak'
                        elif switch == True:
                            change_points_labels.append('Change Point')
                            c = 'r'
                            l = 'Change Point'
                            ICC_curve_list[curve_index]['change_points'][cp_index] = new_index
                        else:
                            change_points_labels.append('Not Change Point')
                            c = 'g'
                            l = 'Not Change Point' 
                        
                        if print_fig == True:
                            ax.plot(case_interp, -np.array(cl_smooth), 'b', lw = 1)
                            ax.set_xlabel('Cumulative Cases (Shifted)')
                            ax.set_ylabel('Loss Statistic Value')
                            ax.plot([self.case_smooth_f[change_point], self.case_smooth_f[change_point]], [0,20], color = c, lw = 1, label = l)
                            # ax.legend()
                            # plt.show()
                    ICC_curve_list[curve_index]['change_points_labels'] = change_points_labels
                    
        else:
            def cp_identification(curve_index, curve, cp_index, change_point):
                try: 
                    if change_point - 3*self.avg_size > curve['change_points'][cp_index - 1]:
                        init = last_param[2:]
                        init[0] = np.amax([1,init[0]])
                    else:

                        init = None
                except:
                    init = None
                case_interp, inc_interp, change_list, last_param = self.window_inference(curve_index, curve, recovery_time_list[curve_index], self.avg_size, peak_or_change = True, change_point = change_point, init = init)
                smooth_size = self.avg_size/2
                cl_smooth = np.empty(len(change_list))
                for index in range(1,len(change_list)):
                    window_size = np.amin([index, smooth_size])
                    cl_smooth[index] = np.sum(change_list[index-int(self.avg_size/2):index+int(self.avg_size/2)])/window_size

                cl_smooth = cl_smooth[6:-6]
                case_interp = case_interp[6:-6]
                change_index = np.argmin(np.array(cl_smooth))
                new_index = np.argmin(np.abs(self.case_smooth_f - case_interp[change_index]))
                check_window = [np.amax([0,len(cl_smooth)/2 - 1*self.avg_size]), len(cl_smooth)/2 + 1*self.avg_size]
                first = cl_smooth[:self.avg_size]
                last = cl_smooth[-self.avg_size:]
                mid = cl_smooth[int(check_window[0]):int(check_window[1])]
                first_err = np.mean(first)
                last_err = np.mean(last)
                side = np.append(first, last)
                var = np.var(side)
                
                # print(check_window, change_index, np.amin(cl_smooth), var, first_err, last_err, np.argmax(inc_interp))
                switch = False
                skip = False
                if np.abs(first_err - last_err) < 1:
                    if np.amin(mid) < np.mean([first_err, last_err])-3:#  np.amin(cl_smooth) > 5*np.sqrt(var):
                        switch = True
                        print('variance')
                    else:
                        skip = True
                elif change_index > check_window[0] and change_index < check_window[-1] and not skip:
                    switch = True
                    print('min')
                if switch == False and np.argmax(inc_interp) > check_window[0] and np.argmax(inc_interp) < check_window[-1] and check_window[0] != 0:
                    change_points_label ='Peak'
                    c = 'purple'
                    l = 'Peak'
                elif switch == True:
                    change_points_label = 'Change Point'
                    c = 'r'
                    l = 'Change Point'
                    ICC_curve_list[curve_index]['change_points'][cp_index] = new_index
                else:
                    change_points_label = 'Not Change Point' 
                    c = 'g'
                    l = 'Not Change Point' 
                    # if print_fig == True:
                    #     ax.plot(case_interp, -np.array(cl_smooth), 'k', lw = 1)
                    #     ax.set_ylim(-10,100)
                    #     ax.plot([self.case_smooth_f[change_point], self.case_smooth_f[change_point]], [0,20], color = c, lw = 1, label = l)
                    #     ax.legend()
                return [change_points_label, case_interp, cl_smooth]
            loop_list = []
            for curve_index, curve in enumerate(ICC_curve_list):
                for cp_index, cp in enumerate(curve['change_points']):
                    loop_list.append([curve_index, curve, cp_index, cp])
            if print_fig == True:
                fig, ax1 = plt.subplots()
                ax1.plot(self.case_smooth_f, self.inc_smooth_f, 'k', lw = 1, label = 'Data')
                ax = ax1.twinx()
            res = Parallel(n_jobs=ncores)(delayed(cp_identification)(curve_index, curve, cp_index, change_point) for curve_index, curve, cp_index, change_point in tqdm(loop_list))
            change_points_labels = np.array(res, dtype = 'object')[:,0]
            case_interp = np.array(res, dtype = 'object')[:,1]
            cl_smooth = np.array(res, dtype = 'object')[:,2]
            print(change_points_labels)
            self.cp_smooth_list.append(cl_smooth)
            total_index = 0
            for curve_index, curve in enumerate(ICC_curve_list):
                for cp_index, cp in enumerate(curve['change_points']):
                    cp_label = change_points_labels[total_index]
                    ICC_curve_list[curve_index]['change_points_labels'].append(cp_label)
                    if print_fig == True:
                        change_point = curve['change_points'][cp_index]
                        if cp_label == 'Not Change Point':
                            c = 'g'
                        if cp_label == 'Peak':
                            c = 'purple'
                        if cp_label == 'Change Point':
                            c = 'r'
                        ax.plot([self.case_smooth_f[change_point], self.case_smooth_f[change_point]], [0,20], color = c, lw = 1)
                        ax.plot(case_interp[total_index], -np.array(cl_smooth[total_index]), 'b', lw = 1)
                        # ax.set_ylim(0,50)
                    total_index += 1
        if fname:
            fig.savefig(fname)
        if overwrite == True:
            self.ICC_curve_list = ICC_curve_list
        
        return ICC_curve_list
                

    def plot_curves(self, title = None, ICC_curve_list = None, full_curve = True, fig_size = (8,6)):
        """
        Plots ICC Curves based on given data.
        
        Parameters
        ----------
        ICC_curve_list : list, default (last list generated in object)
            List of ICC curve dictionaries as generated by "first pass".
            
        Yields
        -------
            Plots of all ICC curves in ICC_curve_list
        """
        if ICC_curve_list == None:
            ICC_curve_list = self.ICC_curve_list  
        if full_curve == False:
            for index, curve in enumerate(ICC_curve_list):
                start_index = curve['start_index']
                shift = curve['shift']
                stop_index = curve['stop_index']
                icc_case = self.case_smooth_f[start_index:stop_index] - shift
                icc_inc = self.inc_smooth_f[start_index:stop_index]
                try:
                    for cp_index, cp_curve in enumerate(curve['curve_obj']):
                        if curve['change_points_labels'] != []:
                            if curve['change_points_labels'][cp_index] == 'Change Point':
                                cp_case = self.case_smooth_f[start_index : curve['change_points'][cp_index]] - shift
                                cp_inc = self.inc_smooth_f[start_index:curve['change_points'][cp_index]]
                                cp_curve.plot_curve(cp_case, inc = cp_inc)
                                start_index = curve['change_points'][cp_index]
                        cp_case = self.case_smooth_f[start_index : stop_index] - shift
                        cp_inc = self.inc_smooth_f[start_index : stop_index]
                        cp_curve.plot_curve(cp_case, inc = cp_inc)
                        start_index = curve['change_points'][cp_index]
                except:
                    fig, ax = plt.subplots()
                    ax.plot(icc_case, icc_inc)
                    ax.set_xlabel('Cumulative Cases')
                    ax.set_ylabel('Incidence')
        else: 
            fig, ax = plt.subplots(figsize = (14,7))
            ax.scatter(self.case_smooth, self.inc_smooth, 3, color = 'k', lw = 1, label = 'Data')
            l = 'ICC Curves'
            for curve_index, curve in enumerate(ICC_curve_list):
                start_index = curve['start_index']
                shift = curve['shift']
                stop_index = curve['stop_index']
                if curve['change_points_labels'] != []:
                    change_index = 0
                    for cp_index, cp_label in enumerate(curve['change_points_labels']):
                        if cp_label == 'Change Point':
                            cp_curve = curve['curve_obj_cp'][change_index]
                            cp_case = self.case_smooth_f[start_index : curve['change_points'][cp_index]] - shift
                            mean = cp_curve.mean(cp_case)
                            ax.plot(cp_case + shift, mean, 'b--', label = l)
                            label = ''
                            start_index = curve['change_points'][cp_index]
                            change_index += 1
                cp_curve = curve['curve_obj_cp'][-1]
                cp_case = self.case_smooth_f[start_index : stop_index] - shift
                mean = cp_curve.mean(cp_case)
                ax.plot(cp_case + shift, mean, 'b--', label = l)
                l = ''
                ax.legend()
                ax.set_ylim([0,1.2*np.amax(self.inc_interp)])
                ax.set_xlabel('Cumulative Cases')
                ax.set_ylabel('Incidence')
                ax.set_title(title)

        return
    def change_dates(self, dates, ICC_curve_list = None, print_res = False):
        """
        Prints the dates associated with the change-points.
        
        Parameters
        ----------
        dates : list
            List of date strings that are associated with inputted data-points.
            
        Returns
        -------
        change_date : list
            List of dates when change points occured.
        """
        if ICC_curve_list == None:
            ICC_curve_list = self.ICC_curve_list        
        change_date = []
        for curve in ICC_curve_list:
            date_dic = {}
            if print_res == True:
                print('Start of Wave : ', dates[curve['start_index']])
            date_dic['start'] = dates[curve['start_index']]
            cp_list = []
            for index, cp in enumerate(curve['change_points']):
                if curve['change_points_labels'][index] == 'Change Point':
                    cp_list.append(dates[cp])
                    if print_res == True:
                        print('Intermediate Change : ', dates[cp])
                if curve['change_points_labels'][index] == 'Peak':
                    cp_list.append(dates[cp])
                    if print_res == True:
                        print('Peak : ', dates[cp])
                if curve['change_points_labels'][index] == 'Not Change Point':
                    cp_list.append('')
            date_dic['intermediate'] = cp_list
            if print_res == True:
                print('End of Wave : ', dates[curve['stop_index']])
            date_dic['stop'] = dates[curve['stop_index']]
        
        
            change_date.append(date_dic)
            curve['dates'] = date_dic
        return change_date
    
    
    def plot_params(self, title = None, ICC_curve_list = None, domain = 'cases', dates =  None):
        """
        Plots parameters along with given ICC curve to illustrate how parameters change from curve to curve
        
        Paramters
        ---------
        ICC_curve_list : list, optional
            List of dictionaries of ICC curves generated by the previous methods. If none is given, most recent ICC_curve_list is used.
        
        domain : str, default : 'cases'
            String ('cases' or 'time'). Denotes the x-axis used when plotting the parameters. 
        """
        if ICC_curve_list == None:
            ICC_curve_list = self.ICC_curve_list
        fig, ax = plt.subplots()
        if domain == 'cases':
            ax.plot(self.case_smooth, self.inc_smooth, 3, color = 'k', lw = 1, label = 'Data', alpha = 0.5)
            ax1 = ax.twinx()
            ax2 = ax1.twinx()
            ax.set_yticks([])
            ax1.tick_params(axis='y', colors='b')
            ax2.set_ylabel('$R_0$', color = 'b')
            ax2.yaxis.set_label_coords(-.1, .5)
            ax2.tick_params(axis='y', colors='g')
            ax1.set_ylabel('$N$', color = 'g')
            ax1.yaxis.set_label_coords(1.1, .5)
            ax.spines['left'].set_color('b')
            ax.spines['right'].set_color('g')
            r = r'$R_0$'
            c = r'$c_\infty$'
            N_cum = 0
            for curve_index, curve in enumerate(ICC_curve_list):
                start_index = curve['start_index']
                shift = curve['shift']
                stop_index = curve['stop_index']
                cp_index = 0
                cp_indices = [i for i, _ in enumerate(curve['change_points_labels']) if _ == 'Change Point']
                if cp_indices != []: 
                    for obj_index, cp_curve in enumerate(curve['curve_obj_cp'][:-1]):
                        cp_index = cp_indices[obj_index]
                        cp_case = self.case_smooth_f[start_index : curve['change_points'][cp_index]] - shift
                        R0 = -np.log(1-cp_curve.h)/cp_curve.h
                        N = cp_curve.c_inf/cp_curve.h
                        # N_cum = N + shift
                        ax1.plot(cp_case + shift, R0*np.ones(len(cp_case)), 'b', label = r, lw = 7, alpha = 0.5)
                        ax2.plot(cp_case + shift, N*np.ones(len(cp_case)), 'g', label = c, lw = 7, alpha = 0.5)
                        r = ''
                        c = ''
                        start_index = curve['change_points'][cp_index]
                cp_curve = curve['curve_obj_cp'][-1]
                cp_case = self.case_smooth_f[start_index : stop_index] - shift
                R0 = -np.log(1-cp_curve.h)/cp_curve.h
                N = cp_curve.c_inf/cp_curve.h
                N_cum += N
                ax1.plot(cp_case + shift, R0*np.ones(len(cp_case)), 'b', label = r, lw = 7, alpha = 0.5)
                ax2.plot(cp_case + shift, N*np.ones(len(cp_case)), 'g', label = c, lw = 7, alpha = 0.5)
                r = ''
                c = ''
                ax.set_ylim([0,1.2*np.amax(self.inc_interp)])
                ax.set_xlabel('Cumulative Cases')
                ax.set_title(title)

        if domain == 'time':
            time = np.arange(len(self.case_smooth))
            ax.plot(time, self.inc_smooth, 3, color = 'k', lw = 1, label = 'Data', alpha = 0.5)
            ax1 = ax.twinx()
            ax2 = ax1.twinx()
            ax.set_yticks([])
            ax1.tick_params(axis='y', colors='b')
            ax2.set_ylabel('$R_0$', color = 'b')
            ax2.yaxis.set_label_coords(-.1, .5)
            ax2.tick_params(axis='y', colors='g')
            ax1.set_ylabel('$N$', color = 'g')
            ax1.yaxis.set_label_coords(1.1, .5)
            ax.spines['left'].set_color('b')
            ax.spines['right'].set_color('g')
            r = r'$R_0$'
            c = r'$c_\infty$'
            N_cum = 0
            for curve_index, curve in enumerate(ICC_curve_list):
                start_index = curve['start_index']
                shift = curve['shift']
                stop_index = curve['stop_index']
                cp_index = 0
                cp_indices = [i for i, _ in enumerate(curve['change_points_labels']) if _ == 'Change Point']
                if cp_indices != []: 
                    for obj_index, cp_curve in enumerate(curve['curve_obj_cp'][:-1]):
                        cp_index = cp_indices[obj_index]
                        cp_case = time[start_index : curve['change_points'][cp_index]]
                        R0 = -np.log(1-cp_curve.h)/cp_curve.h
                        N = cp_curve.c_inf/cp_curve.h
                        # N_cum = N + shift
                        ax1.plot(cp_case, R0*np.ones(len(cp_case)), 'b', label = r, lw = 7, alpha = 0.5)
                        ax2.plot(cp_case, N*np.ones(len(cp_case)), 'g', label = c, lw = 7, alpha = 0.5)
                        r = ''
                        c = ''
                        start_index = curve['change_points'][cp_index]
                cp_curve = curve['curve_obj_cp'][-1]
                cp_case = time[start_index : stop_index]
                R0 = -np.log(1-cp_curve.h)/cp_curve.h
                N = cp_curve.c_inf/cp_curve.h
                N_cum += N
                ax1.plot(cp_case, R0*np.ones(len(cp_case)), 'b', label = r, lw = 7, alpha = 0.5)
                ax2.plot(cp_case, N_cum*np.ones(len(cp_case)), 'g', label = c, lw = 7, alpha = 0.5)
                r = ''
                c = ''
                ax.set_ylim([0,1.2*np.amax(self.inc_interp)])
                ax.set_xlabel('Cumulative Cases')
                ax.set_title(title)
        return

    def window_inference(self, curve_index, curve, Recovery_time, window_size, peak_or_change = False, change_point = None, init = None):
        """
        
        """
        start_values = curve['start_values']
        init_values = curve['init_values']
        start_index = curve['start_index']
        stop_index = curve['stop_index']
        shift = curve['shift']
        window_size = int(window_size)
        c_inf_bounds = start_values[0]
        c0_bounds = start_values[1]
        Recovery_time = [Recovery_time]
        x_win_param_list = []
        x_win_loss_list = []
        x_win_change_list = []
        case = self.case_smooth_f[start_index:stop_index]
        inc = self.inc_smooth_f[start_index:stop_index]
        start_index_interp = np.argmin(np.abs(self.case_interp_prepass - self.case_smooth_f[start_index]))
        stop_index_interp = np.argmin(np.abs(self.case_interp_prepass - self.case_smooth_f[stop_index]))+1
        if case[-1] > self.case_interp_prepass[start_index_interp + window_size]:
            beginning_curve_index = np.where(case > self.case_interp_prepass[start_index_interp + window_size])[0][0]
        else:
            beginning_curve_index = 0

        case_interp = np.concatenate((case[:beginning_curve_index], self.case_interp_prepass[start_index_interp + window_size:stop_index_interp])) - shift
        inc_interp = np.concatenate((inc[:beginning_curve_index], self.inc_interp_prepass[start_index_interp + window_size:stop_index_interp]))
        
        if peak_or_change == True:
            cp_new = np.argmin(np.abs(case_interp - (self.case_smooth_f[change_point] - shift)))
            case_interp = case_interp[np.amax([0,cp_new - int(2*window_size)]) : cp_new + int(2*window_size)]
            inc_interp = inc_interp[np.amax([0,cp_new - int(2*window_size)]) : cp_new + int(2*window_size)]
            
        if not init:
            init = [0,0]
            # p = np.polyfit(case_interp, inc_interp, 2)
            try:
                p = curve_fit(icc, case_interp, inc_interp)[0]
                i=1
                c0_range = np.linspace(-i*shift, case_interp[0])
                while icc(c0_range[0], *p) > 0:
                    i+=1
                    c0_range = np.linspace(-i*shift, case_interp[0])
                    if i == 10:
                        break
                if icc(c0_range[0], *p) < 0:        
                    init[1] = c0_range[icc(c0_range, *p) < 0][-1]
                else:
                    init[1] = np.mean(c0_range)
                i=1.1
                c_inf_range = np.linspace(case_interp[-1], i*case_interp[-1])
                while icc(c_inf_range[-1], *p) > 0:
                    i+=1
                    c_inf_range = np.linspace(case_interp[-1], i*case_interp[-1])
                    if i == 10:
                        break
                if icc(c_inf_range[-1], *p) < 0:
                    init[0] = c_inf_range[icc(c_inf_range, *p) < 0][0]
                else:
                    init[0] = case[-1] #np.mean(c_inf_range)
                # init[0] = case[-1]#np.mean(c_inf_bounds)
                # init[1] = case[0]#np.mean(c0_bounds)
                print('    Case       Inc       init\n', case[0], case[-1], inc[0], inc[-1], init)
            except:
                init = [case[-1], case[0]]
        for case_index, case_value in enumerate(case_interp):
            if case_index < int(window_size/2):
                x_win_param_list.append([0,0,0,0])
                x_win_loss_list.append(np.nan)
                x_win_change_list.append(0)
            if case_index == int(window_size/2):
                start_values = [c_inf_bounds, c0_bounds]

                params = ICC_Find(case_interp[case_index-int(window_size/2):case_index + int(window_size/2)], inc_interp[case_index-int(window_size/2):case_index + int(window_size/2)], Recovery_time, start_values, init_values = init, parallel = False, rand_search_count = 500)[0]
                params_1 = ICC_Find(case_interp[case_index - int(window_size/2) : case_index], inc_interp[case_index - int(window_size/2) : case_index], Recovery_time, start_values, init_values = params[0][2:], parallel = False, rand_search_count = 500)[0]
                params_2 = ICC_Find(case_interp[case_index  : case_index + int(window_size/2)], inc_interp[case_index  : case_index + int(window_size/2)], Recovery_time, start_values, init_values = params[0][2:], parallel = False, rand_search_count = 500)[0]

                win_1_loss = params_1[1]
                win_2_loss = params_2[1]

                x_win_param_list.append(params[0])
                x_win_loss_list.append(params[1])
                x_win_change_list.append(win_1_loss + win_2_loss - params[1])
            elif case_index >  int(window_size/2) and case_index <= len(case_interp) - int(window_size/2):
                c_inf_bounds = [np.amax([8000,case_value]), 1.1*x_win_param_list[-1][2],2]
                c0_bounds = [0.8*x_win_param_list[-1][-1], case_value]
                start_values = [c_inf_bounds, c0_bounds]

                params = ICC_Find(case_interp[case_index-int(window_size/2):case_index + int(window_size/2)], inc_interp[case_index-int(window_size/2):case_index + int(window_size/2)], Recovery_time, start_values, init_values = x_win_param_list[-1][2:], parallel = False, rand_search_count = 100)[0]

                params_1 = ICC_Find(case_interp[case_index - int(window_size/2) : case_index], inc_interp[case_index - int(window_size/2) : case_index], Recovery_time, start_values, init_values = params[0][2:], parallel = False, rand_search_count = 100)[0]
                params_2 = ICC_Find(case_interp[case_index  : case_index + int(window_size/2)], inc_interp[case_index  : case_index + int(window_size/2)], Recovery_time, start_values, init_values = params[0][2:], parallel = False, rand_search_count = 100)[0]

                win_1_loss = params_1[1]
                win_2_loss = params_2[1]

                x_win_param_list.append(params[0])
                x_win_loss_list.append(params[1])
                x_win_change_list.append(win_1_loss + win_2_loss - params[1])
            elif case_index > len(case_interp) - int(window_size/2):
                x_win_param_list.append([0,0,0,0])
                x_win_loss_list.append(np.nan)
                x_win_change_list.append(0)
        if peak_or_change == True:
            return case_interp, inc_interp, x_win_change_list, x_win_param_list[-1]
        else:
            return x_win_param_list, x_win_loss_list, x_win_change_list
    
def loss(x0, case_matrix, Inc_matrix):
    beta = x0[0];  h = x0[1]; c_inf = x0[2]; c0 = x0[3]
    S = 0
    case = case_matrix
    Inc = Inc_matrix
    c = h*case/c_inf
    c0 *= h/c_inf
    kappa = (1-c0)*(1-h)**(-c0/h)

    mean = np.maximum(beta*(case/c_inf-1/np.log(1-h)*np.log(1-c) + 1/np.log(1-h)*np.log(kappa))*(1-c),0)

    one = np.log(kappa/(1-c))/np.log(1-h)
    twoa = h/(np.log(1-h))**2
    twob = (kappa + c - 1)/(1-c)/kappa
    two = twoa*twob

    var = -c_inf*beta**2*(1-c)**2*(one - twoa*twob)
    var = np.maximum(var, 0)

    LS_index = np.where(var<=0)[0]
    WLS_index = np.where(var>0)[0]

    for index in LS_index:
        S += (Inc[index] - c_inf*mean[index])**2
    for index in WLS_index:
        S += 1/(2*var[index])*(Inc[index]-c_inf*mean[index])**2 - 1/2*np.log(2*np.pi*var[index])
    return S
        
def icc(x,a,b,c):
    return -1*np.abs(a)*x**2 + b*x + c
 
    