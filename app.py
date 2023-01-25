import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import math as m
import random as r

#Market information
risk_free = 0.0503

#share specific information
S_0 = 75868633.22  # 10,000,000
sigma = 2.0068
strike = 1000000000
dT=2/365
current_time=0

#European up and out call option information
T = 2
barrier = 1000000000

def terminal_shareprice(S_0, risk_free_rate, sigma, Z, T):
    """
    Generates the terminal share price given some random normal values, z
    """
    # It returns an array of terminal stock prices.
    return S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

def discounted_call_payoff(S_T, K, risk_free_rate, T):
    """
    Function for evaluating the discounted payoff of a call option
    in the Monte Carlo Estimation
    """
    # It returns an array which has the value of the call for each terminal stock price.
    return np.exp(-risk_free_rate*T)*np.maximum(S_T - K, 0)

np.random.seed(0)
num_simulations = 1000
# the number of steps represent the times we simulate the process, with each step comprising
# 1000*i steps so we get simulations from 1000 to 50000
num_steps = 50
num_of_months = 13
# terminal price is an array of size 13 to account for the 12 months plus initial value
# it is timed by the number of steps
term_val = [[None]*num_of_months]*num_steps 

# initialise the monte carlo value, estimates and std as empty array of size number of steps
mbarrier_val = [None]*num_steps
mbarrier_estimates = [None]*num_steps
mbarrier_std = [None]*num_steps

value = 0

for i in range(1,num_steps+1):
    # fill out the first value with our initial stock price
    term_val[i-1][0] = np.full((num_simulations*i), S_0)
    
    for j in range (1,num_of_months):
        # update current month to reflect the monthly simulation we are currently in
        current_month = (j-1)/12
        norm_array = norm.rvs(size = num_simulations*i)
        term_val[i-1][j] = terminal_shareprice(term_val[i-1][j-1],risk_free,sigma,norm_array,dT)
    
    # Compute discounted barrier Price of the option 
    mbarrier_val[i-1] = discounted_call_payoff(term_val[i-1][12],strike,risk_free,T-current_time)
    
    # use the above formula to calculate the values of the barrier option
    ## get array of booleans for when stock is knocked out or not
    knock_in_array = (np.max(term_val[i-1],axis = 0) > barrier)
    ## times it by the value of the previously calculated barrier option
    mbarrier_val[i-1] = mbarrier_val[i-1] * knock_in_array
    
    # compute mean and standard deviation of entire path
    mbarrier_estimates[i-1] = np.mean(mbarrier_val[i-1])
    mbarrier_std[i-1] = np.std(mbarrier_val[i-1]/np.sqrt(i*num_simulations))
    
    newValue = np.max(mbarrier_val[i-1]);
    if (value < newValue):
        value = newValue;

print(value)