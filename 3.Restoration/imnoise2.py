import numpy as np

def imnoise2(type, size, *args):
	M, N = size
	a, b = (0, 1)
	if len(args) in [1, 2]:
		a = args[0]
	if len(args) == 2:
		b = args[1]

	# Noise	with uniform distribution
	if 'uniform' == type:
		r = a + (b - a) * np.random.rand(M, N)
	
	# Noise	with Gaussian (normal) distribution
	elif 'gaussian' == type:
		r = a + b * np.random.randn(M, N)
	
	# 'Salt and Pepper' impulse noise
	elif 'salt & pepper' == type:
		if len(args) == 0:
			a, b = (0.05, 0.05)
			
		if (a + b) > 1:
			raise ValueError('Pa + Pb must not exceed 1')
			
		n = np.random.rand(M, N)
		r = np.full((M, N), 0.5, 'float32')
		r[n <= a] = 0
		u = a + b
		r[np.logical_and(n > a, n <= u)] = 1;
	
	# Noise	with logarithmic normal distribution
	elif 'lognormal' == type:
		if len(args) == 0:
			a, b = (1, 0.25)
		
		r = a * np.exp(b * np.random.randn(M, N))
	
	# Exponential noise
	elif 'exponential' == type:
		if len(args) < 1:
			a = 1
			
		if a < 0:
			raise ValueError('Expected positive parameter')
		
		k = -1 / a
		r = k * np.log(1 - np.random.rand(M, N))
	
	# Rayleigh noise
	elif 'rayleigh' == type:
		r = a + np.sqrt(-b * np.log(1 - np.random.rand(M, N)))
	
	# Erlang noise
	elif 'erlang' == type:
		if len(args) == 0:
			a, b = (2, 5)
		
		k = -1 / a
		r = np.zeros((M, N))
		for i in range(b):
			r = r + k * np.log(1 - np.random.rand(M, N))
	
	else:
		raise ValueError('Unsupported distribution type.')
		
	return r
	