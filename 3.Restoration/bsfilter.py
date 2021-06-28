import numpy as np

def bsfilter(type, M, N, D0, P, n = 1):
	'''Computes frequency domain bandstop or narrowband filters.
       H = bsfilter(type, M, N, D0, W, n) creates the transfer function of
       a band-stop filter, H, of the specified type and size (M-by-N).
       Cutoff frequency is D0.
       In case of bandstop filter P defines bandwidth.
       In case of narrowband filter P defines point to suppress.
         'bs-btw'      Butterworth bandstop filter of order n
         'bs-gaussian' Gaussian bandstop filter
         'nb-btw'      Butterworth narrowband filter of order n
         'nb-gaussian' Gaussian narrowband filter
	'''
	
	# dftuv
	u = np.arange(0, M, 1)
	v = np.arange(0, N, 1)
	u[(M // 2):] = u[(M // 2):] - M
	v[(N // 2):] = v[(N // 2):] - N
	V, U = np.meshgrid(v, u)
	
	if (type[0:2] == 'nb'):
		U = np.roll(U, P, (0, 1))
		V = np.roll(V, P, (0, 1))
	else:
		U = np.fft.fftshift(U)
		V = np.fft.fftshift(V)
		
	D_2 = U ** 2 + V ** 2
	D = np.sqrt(D_2)
	D0_2 = D0 * D0
	
	# Select filter
	if (type == 'bs-btw'):
		H = 1.0 / (1.0 + (D * P / (D_2 - D0_2)) ** (2 * n))
		
	elif (type == 'bs-gaussian'):
		H = 1.0 - np.exp(-0.5 * (((D_2 - D0_2) / (D * P)) ** 2))
		
	elif (type == 'nb-btw'):
		H = 1.0 - 1.0 / (1.0 + (D / D0) ** (2 * n))
		
	elif (type == 'nb-gaussian'):
		H = 1.0 - np.exp(-(D_2) / (2 * D0_2))
		
	return H
	