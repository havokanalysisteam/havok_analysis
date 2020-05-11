'''

      Project: HAVOK analysis
         File: time_series.py
 File Created: 03.03.2020
       Author: Pascal Schulz (pascal.schulz92@icloud.com)
-----
Last Modified: 11.05.2020 21:57:59
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

'''


import numpy as np 


def extract(ts, k, fx = None):
    '''
    ts = time series
    fx = lambda function
    k = column to choose
    '''
    
    dim = ts.shape[1] # find the number of columns
    
    if dim < k:
        
        print('k must be smaller then ', dim)
        
    else:
        # only use the needed column
        #s = np.array(pd.DataFrame(ts)[k])
        s = ts[:, k]
        if fx is not None:      
            # transform the time series
            s = np.asarray([fx(xi) for xi in s])
            
        
        return(s)
    
def get_timeseries(ts, time, tau = 1):
    '''
    This function will return every tau step
    
    ## Later
    - Implementation of approaches that choose tau 
      by him self
    '''
    
    
    
    if isinstance(tau, int):
        # case if tau is an integer
        k = ts.shape[0]
    
        mask = np.arange(0, k, tau)
        
        new_ts = ts[mask]
        new_time = time[mask]
        
        tau_hat = np.diff(new_time)[1] * tau
        
        return(new_ts, tau_hat)
    
    elif type(tau) == np.ndarray:
        # case if tau is a for definded sequence
        tau_hat = np.diff(tau)[1] * tau 
        
        return(ts[tau], tau_hat)
    
    elif isinstance(tau, str):
        # if tau is choosen by a method
        print('Here is some space for improvement :D')
        
