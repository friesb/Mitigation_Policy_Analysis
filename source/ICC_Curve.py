"""
The ICC Curve Object.

This class can be used for basic ICC curve analysis/generation. 

Suggestions for additions should be sent to frieswd@arizona.edu

Author: Bill Fries, GIDP Applied Math, University of Arizona.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import scipy.fft as FFT
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ICC_Curve:
    """
    This is an object which allows for basic storage of determinisitc ICC curves and how they might be used with data.
    
    Attributes
    ----------
    beta : float
        Transmission rate of dissease
    h : float
        Percent of population that becomes infected
    c_inf : float
        Total number of cumulative cases
    c0 : float
        Initial number of cases (may be negative)
    kappa : float
        Percent of population size that is initially susceptible
    mean_curve : ndarray
        Incidence values of the icc curve at given case values
    variance_curve : ndarray
        Variance on incidence values of the icc curve given case values
    case_list : list
        List of case values generated in Markov Chain simulations
    inc_list : list
        List of incidence values generated in Markov Chain simulations
    time_list :
        List of time-stamps generated in Markov Chain simulations
    N : float
        Total population size. Calculated as c_inf/h
    R0 : float
        Basic Reprodoctive number. Calculated as -log(1-h)/h
    gamma : float
        Recovery Rate. Calculated as beta/R0 or 1/Recovery Time
    traj : ndarray
        Deterministic trajectory generated by scipy.integrate.odeint.
    case_list : list
        List of lists of case values generated by the Markov Chain simulations
    inc_list : list
        List of lists of incidence values generated by the Markov Chain simulations.
    time_list : list
        List of lists of time steps generated by the Markov Chain simulations.
    Methods
    -------
    mean(case, scaled = False)
        Solves mean incidence curve given case values
    
    variance(case, scaled = False)
        Solves theoretical variance curve given case values
    
    loss_to_data(case, inc)
        Finds loss value of loss function given case and incidence values
    
    MC_simulations(initial_condition, trials = 1)
        Runs Markov Chain simulations
    
    plot_curve(case, inc = None, var = False)
        Plots mean curve, incidence and variance if desired.
    
    time_integration(deterministic = True, time = None, stochastic = False)
        Generates compartmental model trajectories and plots them, potentially with Markov Chain simulations if desired.
    
    """
    def __init__(self, params):
        self.beta = params[0]
        self.h = params[1]
        self.c_inf = params[2]
        self.c0 = params[3]*self.h/self.c_inf
        self.R0 = -np.log(1-self.h)/self.h
        self.gamma = self.beta/self.R0
        self.N = self.c_inf/self.h
        
        return
    
    def mean(self, case,  scaled = False):
        """
        Parameters
        ----------
        case : ndarray
            monontonic array of cumulative case values.
            
        scaled : bool, optional
            Scales mean curve by c_inf. (curve ends at 1).
            
        Returns
        -------
        mean_curve : ndarray
            mean incidence values
        """
        c = self.h*case/self.c_inf
        self.kappa = (1-self.c0)*(1-self.h)**(-self.c0/self.h)       
        self.mean_curve = self.beta*(case/self.c_inf-1/np.log(1-self.h)*np.log(1-c) + 1/np.log(1-self.h)*np.log(self.kappa))*(1-c)
        
        if scaled == False:
            self.mean_curve *= self.c_inf
        return self.mean_curve
    
    def variance(self, case, scaled = False):
        """
         Parameters
        ----------
        case : ndarray
            monontonic array of cumulative case values.
            
        scaled : bool, optional
            Scales variance curve by c_inf. (men curve ends at 1).
            
        Returns
        -------
        variance_cuve : ndarray
            variance on incidence values       
        """
        c = self.h*case/self.c_inf
        one = np.log(self.kappa/(1-c))/np.log(1-self.h)
        twoa = self.h/(np.log(1-self.h))**2
        twob = (self.kappa + c - 1)/(1-c)/self.kappa
        two = twoa*twob
        self.variance_curve = -self.c_inf*self.beta**2*(1-c)**2*(one - twoa*twob)
        
        if scaled == True:
            self.variance_curve /= self.c_inf
        self.variance_curve = np.maximum(self.variance_curve, 0)    
        return self.variance_curve
    
    def loss_to_data(self, case_list, inc):
        """
         Parameters
        ----------
        case : ndarray
            monontonic array of cumulative case values.
            
        inc : ndarray
            incidence values for given cumulative cases
            
        Returns
        -------
        S : float
            sum of individual loss values using log-liklihood estimator       
        """
        S = 0
        try:
            test = len(inc[0])
            for data_index, inc_ind in enumerate(inc):
                case = case_list[data_index]
                c = self.h*case/self.c_inf
                self.mean(case)
                self.variance(case)

                self.variance_curve = np.maximum(self.variance_curve, 0)
                
                LS_index = np.where(self.variance_curve<=1/2/np.pi)[0]
                WLS_index = np.where(self.variance_curve>1/2/np.pi)[0]
                if np.amax(self.mean_curve) <= 0 or np.isnan(np.amin(self.mean_curve)) == True:
                    print('Warning: The given case data does not correspond to the parameters of the ICC curve. Mean < 0')
                    return
                else:
                    for index in LS_index:
                        S += (inc_ind[index] - self.mean_curve[index])**2
                    for index in WLS_index:
                        S += 1/(2*self.variance_curve[index])*(inc_ind[index] - self.mean_curve[index])**2 - 1/2*np.log(2*np.pi*self.variance_curve[index])
        except:
            case = case_list
            c = self.h*case/self.c_inf
            self.mean(case)
            self.variance(case)

            self.variance_curve = np.maximum(self.variance_curve, 0)
            LS_index = np.where(self.variance_curve<=1/2/np.pi)[0]
            WLS_index = np.where(self.variance_curve>1/2/np.pi)[0]
            if np.amax(self.mean_curve) <= 0 or np.isnan(np.amin(self.mean_curve)) == True:
                print('Warning: The given case data does not correspond to the parameters of the ICC curve. Mean < 0')
                return
            else:
                for index in LS_index:
                    S += (inc[index] - self.mean_curve[index])**2
                for index in WLS_index:
                    S += 1/(2*self.variance_curve[index])*(inc[index] - self.mean_curve[index])**2 - 1/2*np.log(2*np.pi*self.variance_curve[index])
        return S
    
    def MC_simulation(self, trials=1, time_sampling = True, sampling_rate = 1, betaSI = True, variable_N = False):
        """
        This simulates the Markov Chain with the given parameters using the Gillespie Algorithm. This simulates the pure-birth process in cases. The method outputs either a list of case/incidence values for each event or a time-sampled trajectory for a specified time-step (ie. 1 day as default). 
        
        Parameters
        ----------
        trails : int, default 1
            Number of simulations performed.
        time_sampling : bool, default True
            Boolean to determine if time-sampling is performed.
        sampling_rate : int, default 1
            Frequency of time-sampling. 1 -> 1 time-scale (typically 1 day).
        betaSI : bool, default True
            Boolean to determine if the theoretical incidence (\beta SI) or finite difference incidence is used.
        Variable_N : bool, default False
            Boolean to determine if a change in N occurs or not. (STILL WORK IN PROGRESS)
            
        Returns
        -------
        case_list : list
            List of lists of case values for each event. One list for each trial.
        inc_list : list
            List of lists of incidence values for each event. One list for each trial.
        """
        self.case_list = []
        self.inc_list = []
        self.time_list = []

        beta = self.beta/self.N
        init = -int(self.c0*self.c_inf/self.h)#self.mean(np.array([0])))
        total_prob = beta*(self.N-init)*init

        no_change = False
        for _ in range(trials):
            total_prob = beta*(self.N-init)*init
            CaseCount = []
            RecCount = []
            t_list = []
            case = []
            Inc = []
            CaseCount.append([0, init])
            RecCount.append(0)
            Cases = init
            N = self.N
            Rec = 0
            while Cases>Rec:
                t = -np.log(np.random.rand(1))/total_prob
                t_list.append(t)
                lambda_C = self.R0*(N-Cases)/N/(self.R0*(N-Cases)/N+1)
                pull = np.random.rand(1)
                if pull < lambda_C:
                    Cases += 1
                    total_prob = beta*(N-Cases)*(Cases-Rec) + self.gamma*(Cases-Rec)
                else:
                    Rec += 1
                    total_prob = beta*(N-Cases)*(Cases-Rec) + self.gamma*(Cases-Rec)
                CaseCount.append([t[0], Cases])
                RecCount.append(Rec)
                if variable_N == True:
                    Change_N_stat = 0.000001 * (Cases)/N
                    if self.N/2 < Cases and Cases < self.N: #np.random.rand(1) < Change_N_stat:
                        N = self.N * ((np.tanh((Cases - self.N*0.75)/(self.N*.08)) + 1)*2 + 1)
                        beta = self.beta/N
                        total_prob = beta*(N-Cases)*(Cases-Rec) + self.gamma*(Cases-Rec)
                        no_change = True
                if Cases - Rec <=0:
                    break
            case_start = np.array(CaseCount)[:,1]
            RecCount = np.array(RecCount)
            if time_sampling == False:
                case.append(case_start[::sampling_rate])
                Inc.append(beta*(self.N - case_start[::sampling_rate])*(case_start[::sampling_rate]-RecCount[::sampling_rate]))
                self.time_list.append(np.cumsum(t_list))
            else:
                day_count = sampling_rate
                T = np.cumsum(t_list)
                day = 0
                k=0
                last = 0
                inc_day = []

                StateCount = case_start
                Rec_count = 0
                Rec_count_last = 0
                beta = self.beta/self.N
                N = self.N
                no_change = False
                for ii in range(2,int(T[-1]/day_count)+1):
                    count = 0
                    while T[k] <= day+day_count :
                        if k == len(T)-1:
                            break
                        k += 1
                        count += 1
                        if StateCount[k] == StateCount[k-1]:
                            Rec_count += 1
                    if k == len(T)-1:
                        break
                    else:
                        if self.N/2 < StateCount[k] and StateCount[k] < self.N and variable_N: #np.random.rand(1) < Change_N_stat:
                            N = self.N * ((np.tanh((StateCount[k] - self.N*0.75)/(self.N*.08)) + 1)*2 + 1)
                            beta = self.beta/N
                        inc_day.append((StateCount[k]-StateCount[last])/(T[k]-T[last]))
                        Inc.append((beta*(N-StateCount[k])*(StateCount[k]-Rec_count)+beta*(N-StateCount[last])*(StateCount[last]-Rec_count_last))/2)
                        case.append((StateCount[k]+StateCount[last])/2)
                        last += count
                        day += day_count
                        Rec_count_last = Rec_count
                case = np.array(case)
                if betaSI:
                    Inc = np.array(Inc)
                else:
                    Inc = np.array(inc_day)
            self.case_list.append(case)
            self.inc_list.append(Inc)
            self.time_list.append(np.arange(len(case)))
        return self.case_list, self.inc_list

    def plot_curve(self, case_list, inc = "", var = False):
        """
        Parameters
        ----------
        case : ndarray
            monontonic array of cumulative case values.
            
        inc : ndarray, optional
            incidence values for given cumulative cases
            
        Yields
        -------
        figure : 
            plots of mean curve, incidence (optional) and standard deviation (optional)
        """
        fig, ax = plt.subplots()
        try:
            test = len(case_list[0])
            case = [0]
            for case_test in case_list:
                if case_test[-1] > case[-1]:
                    case = case_test
        except:
            case = case_list
        ax.plot(case, self.mean(case),'b', linewidth = 1, label = 'ICC Curve')
            
        ax.set_xlabel('Cumulative Cases')
        ax.set_ylabel('Incidence')
        if var:
            case_var = np.arange(0, 1.3*case[-1])
            ax.plot(case_var, self.mean(case_var) + np.sqrt(self.variance(case_var)), 'r--', linewidth = 1, label = 'Standard Deviations')
            ax.plot(case_var, self.mean(case_var) - np.sqrt(self.variance(case_var)), 'r--', linewidth = 1)
            ax.plot(case_var, self.mean(case_var) + 2*np.sqrt(self.variance(case_var)), 'r--', linewidth = 1, alpha = 0.66)
            ax.plot(case_var, self.mean(case_var) - 2*np.sqrt(self.variance(case_var)), 'r--', linewidth = 1, alpha = 0.66)
            ax.plot(case_var, self.mean(case_var) + 3*np.sqrt(self.variance(case_var)), 'r--', linewidth = 1, alpha = 0.33)
            ax.plot(case_var, self.mean(case_var) - 3*np.sqrt(self.variance(case_var)), 'r--', linewidth = 1, alpha = 0.33)
        if type(inc) != str:
            label = 'Data'
            try:
                test = len(inc[0])
                trials = len(inc)
                a = 1
                for index, inc_ind in enumerate(inc):
                    ax.scatter(case_list[index], inc_ind, marker = 'x', color = 'k', s = 3, lw = 1, alpha = a, label = label)
                    label = ''
                    a = np.amin([1,10/trials])
            except:
                ax.scatter(case, inc, s = 1, marker = 'x', color = 'k', label = label, lw = 1)
        try:
            ax.legend(title = r'N : {:.0f} | $R_0$ : {:.2f}'.format(self.N, self.R0))
        except:
            ax.legend()
        ax.set_ylim(0, 1.5*np.amax(self.mean(case)))
        plt.show()
        return
    
    def time_integration(self, deterministic = True, time = None, stochastic = False):
        """
        Generates the SIR trajectories for the given curve.
        
        Parameters
        ----------
        deterministic : bool, default True
            Boolean to determine if deterministic trajectories are generated and plotted.
        time : list, default None
            Required if determinisitic = True. Time steps for odeint to integrate over.
        stochastic : bool, default False
            Boolean to determine if Markov Chain simulations are plotted. This requires MC_simulations to be run first. 
            
        Yields
        ------
        self.traj : ndarray
            If deterministic = True, self.traj is the resulting trajectory from odeint.        
        """
        fig, ax = plt.subplots()
        
        ax.set_xlabel('People')
        ax.set_ylabel('Time')
        s_label = 'Susceptible'
        i_label = 'Infectious'
        r_label = 'Removed'
        if deterministic == True:
            try:
                def sir(x,t):
                    s = x[0]; i = x[1]
                    sdot = -self.beta*s*i/self.N
                    idot = self.beta*s*i/self.N - self.gamma*i
                    return [sdot, idot]
                self.traj = odeint(sir, [self.N, self.N-self.kappa*self.N], time)

                ax.plot(time, self.traj[:,0], 'b', lw = 5, label = s_label, alpha = 0.5)
                ax.plot(time, self.traj[:,1], 'r', lw = 5, label = i_label, alpha = 0.5)
                ax.plot(time, self.N  - np.sum(self.traj, axis = -1), 'g', lw = 5, label = r_label, alpha = 0.5)
                s_label = ''
                i_label = ''
                r_label = ''
            except:
                print('Time is not defined for odeint. Consider time = np.arange(0,100)')
                return
            
        if stochastic == True:
            try:
                for sim_index, case in enumerate(self.case_list):
                    case = case[0]
                    s_list = []
                    i_list = []
                    s = self.N - case[0]
                    i = case[0]
                    time = self.time_list[sim_index]
                    for t_index, t in enumerate(time):
                        if case[t_index + 1] > case[t_index]:
                            i += 1
                            s -= 1
                        else:
                            i -= 1
                        s_list.append(s)
                        i_list.append(i)
                    r_list = self.N*np.ones(len(s_list)) - s_list - i_list
                    ax.plot(time, s_list, 'b:', lw = 1, label = s_label)
                    ax.plot(time, i_list, 'r:', lw = 1,label = i_label)
                    ax.plot(time, r_list, 'g:', lw = 1, label = r_label)
                    s_label = ''
                    i_label = ''
                    r_label = ''
            
            except:
                print('If stochasitic = True, you must first run MC_simulations with time_sampling = False')
                return
        ax.legend(title = r'N : {:.0f} | $R_0$ : {:.2f}'.format(self.N, self.R0))
        return