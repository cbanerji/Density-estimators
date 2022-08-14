# generate synthetic dataset
import numpy as np
from scipy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math
import itertools
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

# Generate synthetic data for density estimation
def gen_data(k=3, dim=1, points_per_cluster=500, lim=[-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    '''
    x = []
    cove = []
    mean = random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    for i in range(k):
        cov = random.rand(dim, dim+10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
        cove.append(cov)
    x = np.clip(np.array(x),-10,10)
    return x, np.squeeze(mean), np.squeeze(cove) #return data, params. of distr.

# Define Gaussian Kernel function
def gauss(j, m, h):
   j = float(m-j) / h
   return math.exp(-j*j/2.0) / math.sqrt(2.0*math.pi) / h

# Calculate KDE
def cal_den(x, h, a_val):
    den = []
    n = x.size
    for j in a_val:
        sm = []
        for m in x:
            gauss_val = gauss(j,m,h)
            sm.append(gauss_val)
        sm_sum = sum(sm)
        kde = (1/(n*h))*sm_sum
        den.append(kde)
    fin_den.append(den)
    return fin_den


if __name__ == "__main__":
    fin_den = []
    h  = 0.75
    x,mu,co = gen_data() #obtain synthetic data
    a_val = np.linspace(np.min(x),np.max(x), len(x)) #generate ddatapoints for kde cal.
    dnsi = cal_den(x, h, a_val) #Calculate kde, for all datapoints in 'a_val'
    dnsi = [dnsi for sublist in dnsi for dnsi in sublist] #Flatten

    #--Plotting--
    '''
    # plot data generating distributions
    val = np.arange(np.min(x),np.max(x), 0.5)
    for t in range(3):
        plt.plot(val, norm.pdf(val, mu[t], co[t]))
    '''
    plt.hist(x, density = True, bins = 100) #plot normed histogram of data
    plt.plot(a_val, dnsi, '-k', label='KDE density') #plot kde density estimate
    plt.legend()
    plt.show()
