#!/usr/bin/env python

from scipy import *
import pylab as py

def vanilla(r, K, dt, sigma, S):
    """
    This is a simple Vanilla Put Option calcualtion
    based on the analytic solution for a single
    underlying asset. The solution used is from
    The Mathematics of Financial Derivatives, Wilmott, et al.
    Uses ndtr and exp from scipy and ndtr scipy.special modules.
    """
    
    d1 = zeros(len(S))
    d2 = zeros(len(S))
    n1 = zeros(len(S))
    n2 = zeros(len(S))    
    pt = zeros(len(S))
    zpt= zeros(len(S))
    b = sigma*sqrt( dt)
    dsct = exp(-1.0*r*dt)
    for i in range(len(S)):
        d1[i] = (log(S[i]/K) + (r + (0.5* sigma**2))*dt)/b
        d2[i] = (log(S[i]/K) + (r - (0.5* sigma**2))*dt)/b
        n1[i] = special.ndtr(-1.0*d1[i])
        n2[i] = special.ndtr(-1.0*d2[i])    
        pt[i] = 1.0*K*dsct*n2[i] - S[i]*n1[i]
        zpt[i]= max(K-S[i],0) # PUT At expiry
        ## zpt[i]= max(S[i]-K,0) # CALL At expiry


    return pt, zpt

print vanilla.__doc__

r = 0.06
k = 60.0
dt = 0.25
S = linspace(50.0,70.0,41)
sig = 0.2
put,z = vanilla(r,k,dt,sig,S)

diff = put - z
py.plot(S,put,'b-',S,z,'r-',S,diff,'g-')
py.title('Single Underlying Put Option 3 Months to Expiry')
py.xlabel('Underlying Current Value')
py.ylabel('Value')
py.legend(('Put Value','Intrinsic Value','Time Value'),loc='upper right')
py.grid()
py.show()

#for i in range(len(S)):
#    print S[i], put[i], diff[i] 
