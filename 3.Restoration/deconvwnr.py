import numpy as np

def deconvwnr(image, psf, nsr = 0):
	''' Perform Wiener filtering:
	    image - image to filtering
		psf - point spread function
		nsr - noise-to-signal ratio
	'''
	G = np.fft.fft2(image)

	# Shifting the center of psf to the (0, 0) coordinates 
	p = np.zeros(G.shape)
	p[0:psf.shape[0], 0:psf.shape[1]] = psf
	p = np.roll(p, (-(psf.shape[0] - 1) // 2, -(psf.shape[1] - 1) // 2), (0, 1))
	H = np.fft.fft2(p)

	T = np.abs(H) ** 2 + np.finfo(H.dtype).eps
	F = (G / (H + np.finfo(H.dtype).eps) * T / (T + nsr))
	f = np.fft.ifft2(F).real
	
	return f
