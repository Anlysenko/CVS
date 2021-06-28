import numpy as np
import scipy.ndimage.filters as filters

def adpmedian(g, Smax):
    
    if (Smax <= 1) or (Smax / 2 == np.round(Smax / 2)) or (Smax != np.round(Smax)):
        print('SMAX must be an odd integer > 1.')
    M, N = np.shape(g)
    #Initial setup.    
    f = g
    f[:] = 0
    alreadyProcess = false(np.shape(g))

    for k in range(3, 2, Smax):
        zmin = filters.rank_filter(g, rank = 1, size = np.ones([k, k]))
        zmax = filters.rank_filter(g, rank = (k * k), size = np.ones([k, k]))
        zmed = filters.median_filter(g, size = (k, k))

        processUsingLevelB = (zmed > zmin) and (zmax > zmed) and not alreadyProcessed
        zB = (g > zmin) and (zmax > g)
        outputZxy  = processUsingLevelB and zB
        outputZmed = processUsingLevelB and not zB
        f[outputZxy] = g[outputZxy]
        f[outputZmed] = zmed[outputZmed]
   
        alreadyProcessed = alreadyProcessed or processUsingLevelB
        if all(alreadyProcessed[:]):
             break

    f[not alreadyProcessed] = zmed[not alreadyProcessed]
    return f

    