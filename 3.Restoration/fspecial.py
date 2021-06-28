import numpy as np
import scipy.ndimage.filters as filters
from scipy import signal

def gaussian(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def adpmedian(g, Smax, iteration):
    """
    ADPMEDIAN Perform adaptive median filtering.
    ADPMEDIAN(G, SMAX) performs adaptive median filtering of
    image G.  The median filter starts at size 3-by-3 and iterates up
    to size SMAX-by-SMAX. SMAX must be an odd integer greater than 1.

    SMAX must be an odd, positive integer greater than 1.
    iteration must be positive integer no less than 1.
    """
    if np.logical_or(np.logical_or(Smax <= 1, Smax / 2 == np.round(Smax / 2)), Smax != np.round(Smax)):
        print('SMAX must be an odd integer > 1.')
    M, N = np.shape(g)
    #Initial setup
    f = g
    f = np.zeros(np.shape(f))
    alreadyProcessed = np.zeros(np.shape(g))
    
    for k in range(3, Smax, 2):
      zmin = signal.order_filter(g, np.ones((k, k)), 1)
      zmax = signal.order_filter(g, np.ones((k, k)), (k*k)-1)
      zmed = signal.medfilt2d(g, (k, k))

      processUsingLevelB = np.logical_and(np.logical_and(zmed > zmin, zmax > zmed),  np.logical_not(alreadyProcessed))
      zB =  np.logical_and(g > zmin, zmax > g)
      outputZxy  = np.logical_and(processUsingLevelB, zB)
      outputZmed =  np.logical_and(processUsingLevelB,  np.logical_not(zB))
      f[outputZxy] = g[outputZxy]
      f[outputZmed] = zmed[outputZmed]
  
      alreadyProcessed =  np.logical_or(alreadyProcessed, processUsingLevelB)
      if np.all(alreadyProcessed):
        break

    f[np.logical_not(alreadyProcessed)] = zmed[np.logical_not(alreadyProcessed)]
    if iteration == 1:
      return f
    else:
      return adpmedian(f, Smax, iteration-1)

    