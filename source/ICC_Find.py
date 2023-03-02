import numpy as np
import warnings
warnings.filterwarnings('ignore')
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm, trange

def ICC_Find(case, Inc, Recovery_time, start_ranges, parallel = False, rand_search_count = 1000, init_values = []):
    """
    This Function is used to find the best fit for an ICC curve through random search. 
    
    Parameters
    ----------
        case : list
            This is the cumulative case data for the desired ICC Curve.
        Inc : list
            This is the incidence data for the desired ICC Curve.
        Recovery_time : list
            List of #'s. Potential recovery times for the disease. Used to calculate \gamma
        start_ranges : list
            List of c_inf bounds and C_0 bounds. 
        parallel : bool, default False
            If parallel is desired. Uses joblib and runs random search on 2 cores. (Not fully debugged)
        rand_search_count : int, default 1000
            Number of random searches to be done. For larger areas, use larger values
        init_values : list, optional
            If entered, list of initial c_inf and kappa values. Otherwise, initialized with random points in the given ranges.
    
    Returns
    -------
        x_coarse : list
            [[Optimal Parameters for Recovery_time[0]], associated loss]
             [Optimal Parameters for Recovery_time[1]], associated loss]
                        ...
             [Optimal Parameters for Recovery_time[-1]], associated loss]]
    """

    
    def loss(x0, case_matrix, Inc_matrix, print_res = False):
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
            LS_index = np.where(var >= np.amax(c_inf*mean)*0.5)[0]
            WLS_index = np.where(var < np.amax(c_inf*mean)*0.5)[0]
            if np.amax(mean) <= 0 or np.isnan(np.amin(mean)) == True:
                S = 10e10
            else:
                for index in LS_index:
                    S += (Inc[index] - c_inf*mean[index])**2
                for index in WLS_index:
                    S += 1/(2*var[index])*(Inc[index]-c_inf*mean[index])**2 # - 1/2*np.log(2*np.pi*var[index])
            return S
    c_inf_range = start_ranges[0]
    c0_range = start_ranges[1]
    
    if c_inf_range[0] <= 0:
        c_inf_range[0] = 1
    try:
        len(Recovery_time)
    except:
        Recovery_time = [Recovery_time]
    
    x_coarse = []
    for Rec_time in Recovery_time : #tqdm(Recovery_time, position = 1, desc = "Gamma", leave = False):
        gamma = 1/Rec_time
        trough_points_list = []
        trough_loss_list = []
        
        # Initialize best c_inf and kappa from init_values or randomly
        loss_list = []
        h = 0.99
        beta = -gamma*np.log(1-h)/h 
        loss_list = np.empty([50,50])
        for cinf_index, c_inf in enumerate(np.linspace(c_inf_range[0], c_inf_range[1],50)):
            for c0_index, c0 in enumerate(np.linspace(c0_range[0], c0_range[1],50)):
                loss_list[cinf_index, c0_index] = loss([beta, h, c_inf, c0], case, Inc)
            
        best = np.where(loss_list == np.amin(loss_list))
        best_cinf = np.linspace(c_inf_range[0], c_inf_range[1],100)[best[0][0]]
        best_c0 = np.linspace(c0_range[0], c0_range[1],50)[best[1][0]]
                 
        if init_values != []:
            loss_list = []
            c_inf = init_values[0]
            c0 = init_values[1]
            loss_list.append(loss([beta, h, c_inf, c0], case, Inc))
            loss_list.append(loss([beta, h, best_cinf, c0], case, Inc))
            loss_list.append(loss([beta, h, c_inf, best_c0], case, Inc))
            loss_list.append(loss([beta, h, best_cinf, best_c0], case, Inc))
            best = np.argmin(loss_list)
            if best == 0 or best == 2:
                best_cinf = c_inf
            if best == 0 or best == 1:
                best_c0 = c0

        trough_points = []
        trough_loss = []
        n_cores = 4
        h_start = .1
        best_cinf_counter = 0
        
        if init_values != []:
            c_inf_center = init_values[0]
        else:
            c_inf_center = np.mean(c_inf_range)
            
        h_1 = h_start
        h_list = np.linspace(h_1, .99, 100)
        beta_list = np.empty(100)
        for i,h in enumerate(h_list):
            beta_list[i] = -gamma*np.log(1-h)/h
        c_inf_range_scale = [c_inf_range[0]/c_inf_center, c_inf_range[1]/c_inf_center]
        if parallel == False:
            for index in range(rand_search_count):
                starting_values = np.random.uniform(np.array([c0_range[0], c_inf_range_scale[0]]), np.array([c0_range[1], c_inf_range_scale[1]]), (1,2))
                c_inf = starting_values[0,1]*c_inf_center
                c0 = starting_values[0,0]
                h_loss_list = []
                loss_matrix = np.empty([100,4])
                for index, h in enumerate(h_list):
                    beta = beta_list[index] 
                    loss_matrix[index,0] = loss([beta, h, c_inf, c0], case, Inc)
                    loss_matrix[index, 1] = loss([beta, h, best_cinf, c0], case, Inc)
                    loss_matrix[index, 2] = loss([beta, h, c_inf, best_c0], case, Inc)
                    loss_matrix[index, 3] = loss([beta, h, best_cinf, best_c0], case, Inc)

                best_loss_index = np.where(loss_matrix == np.amin(loss_matrix))
                h = h_list[best_loss_index[0]][0]
                beta = -gamma*np.log(1-h)/h

                if np.isin(0, best_loss_index) or np.isin(2, best_loss_index): # Update best c_inf if necessary
                    best_cinf = c_inf
                    best_cinf_counter = 0
                else:
                    best_cinf_counter += 1
                    if best_cinf_counter == c_inf_range[1]/100:
                        c_inf_range = [0.99*best_cinf, 1.01*best_cinf] # Refine search area if necessary

                if np.isin(0, best_loss_index) or np.isin(1, best_loss_index): # Update kappa if necessary
                    best_c0 = c0

                trough_points.append([beta, h, best_cinf, best_c0])
                trough_loss.append(loss([beta, h, best_cinf, best_c0], case, Inc))
        else:
            def Random_Search(index):
                kappa, c_inf = starting_values[index]
                h_1 = h_start
                h_list = np.linspace(h_1, .99, 100)
                h_loss_list = []
                h_index = 0
                c_inf_temp = []
                kappa_temp = []

                loss_matrix = np.empty([100,4])
                for index, h in enumerate(h_list):
                    beta = -gamma*np.log(1-h)/h
                    loss_matrix[index,0] = loss([beta, h, c_inf, kappa], case, Inc)
                    loss_matrix[index, 1] = loss([beta, h, best_cinf, kappa], case, Inc)
                    loss_matrix[index, 2] = loss([beta, h, c_inf, best_kappa], case, Inc)
                    loss_matrix[index, 3] = loss([beta, h, best_cinf, best_kappa], case, Inc)

                best_loss_index = np.where(loss_matrix == np.amin(loss_matrix))
                h = h_list[best_loss_index[0]][0]
                beta = -gamma*np.log(1-h)/h

                if np.isin(0, best_loss_index) or np.isin(2, best_loss_index):
                    best_cinf = c_inf
                if np.isin(0, best_loss_index) or np.isin(1, best_loss_index):
                    best_kappa = kappa

                return trough_points, trough_loss

            trough_points, trough_loss = Parallel(n_jobs=2)(delayed(Random_Search)(index) for index in trange(len(starting_values)))

        index = np.argmin(trough_loss)
        x_coarse.append([trough_points[index], trough_loss[index]])
                        
    return x_coarse

