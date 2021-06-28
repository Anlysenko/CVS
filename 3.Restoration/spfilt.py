import numpy as np
import scipy.ndimage.filters as filters

def spfilt(g, type, m, n, param = None):
	if ('min' == type):
		f = filters.minimum_filter(g, size = (m, n))
		
	elif ('max' == type):
		f = filters.maximum_filter(g, size = (m, n))
		
	elif ('midpoint' == type):
		f = ((filters.minimum_filter(g, size = (m, n)).astype('uint32') +
		     filters.maximum_filter(g, size = (m, n)).astype('uint32')) / 2).astype(g.dtype);
		
	elif ('amean' == type):
		h = np.ones((m, n)) / (m * n)
		f = filters.convolve(g, h, mode = 'nearest')
		
	elif ('gmean' == type):
		h = np.ones((m, n))
		f = np.log(g.astype('float32') + np.finfo('float32').eps)
		f = (np.exp(filters.convolve(f, h)) ** (1 / (m * n))).astype(g.dtype)
		
	elif ('hmean' == type):
		h = np.ones((m, n))
		f = 1.0 / (g.astype('float32') + np.finfo('float32').eps) 
		f = ((m * n) / filters.convolve(f, h)).astype(g.dtype)
		
	elif ('chmean' == type):
		if (param == None):
			Q = 1.5
		else:
			Q = param
			
		h = np.ones((m, n))
		f = g.astype('float32') + np.finfo('float32').eps
		f = (filters.convolve(f ** (Q + 1), h) / (filters.convolve(f ** Q, h))).astype(g.dtype)
	
	elif ('atrimmed' == type):
		if (param == None):
			d = 2
		else:
			d = param
		
		h = np.ones((m, n))
		f = filters.convolve(g.astype('float32'), h)
		
		for k in range(d // 2 - 1):
			f = f - filters.rank_filter(g, rank = k, size = (m, n))
			
		for k in range(m * n - (d // 2), m * n):
			f = f - filters.rank_filter(g, rank = k, size = (m, n))

		f = np.clip((f / (m * n - d)), 0, 255).astype(g.dtype)
	
	elif ('median' == type):
		f = filters.median_filter(g, size = (m, n))
	
	else:
		print('Unsupported filter type: {}, '.format(type))
		
	return f