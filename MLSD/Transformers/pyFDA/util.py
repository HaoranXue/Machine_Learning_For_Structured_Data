import numpy as np
import scipy.interpolate

def binarySearch_inverseMonotonic(t0,t1,f,ft,i):
	"""Binary search on a monotone increasing function for the inverse of f at point t"""
	tm = (t0+t1)/2
	if i > 100:
		return tm
	f0 = f(t0) #scipy.interpolate.splev(t0,f)
	f1 = f(t1) #scipy.interpolate.splev(t1,f)
	if np.abs(f0-ft) < 1e-3:
		return t0
	if np.abs(f1-ft) < 1e-3:
		return t1
	fm = f(tm)#scipy.interpolate.splev(tm,f)
	if fm < ft:
		return binarySearch_inverseMonotonic(tm,t1,f,ft,i+1)
	else:
		return binarySearch_inverseMonotonic(t0,tm,f,ft,i+1)
