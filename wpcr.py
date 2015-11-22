import numpy
import scipy.signal
from matplotlib.pylab import *

tau = numpy.pi * 2

# find period of fundamental frequency
# input: magnitude spectrum (abs(fft()) or power spectrum
# output: approximate samples per symbol of fundamental frequency
def find_fundamental(spectrum):
    ac = numpy.array(abs(scipy.fft(spectrum)))
    ac[0] = 0
    ac[1] = 0
    ac[-1] = 0
    maxima = scipy.signal.argrelextrema(ac, numpy.greater_equal)[0]
    i = maxima[matplotlib.pylab.find(ac[maxima] > max(ac)*0.4)[0]]
    estimate = 1.0*((i-1)*ac[i-1] + i*ac[i] + (i+1)*ac[i+1]) / (ac[i-1] + ac[i] + ac[i+1])
    return estimate

# find spectral peak closest to guessed frequency
# input: magnitude spectrum (abs(fft()) or power spectrum, bin number guess
# output: bin number of peak near guess
def find_peak(spectrum, guess):
    maxima = scipy.signal.argrelextrema(spectrum, numpy.greater_equal)[0]
    return maxima[(abs(maxima - guess)).argmin()]

# whole packet clock recovery
# input: real valued NRZ-like waveform (array, tuple, or list)
#        must have at least 2 samples per symbol
#        must have at least 3 symbols
# output: list of symbols
def wpcr(a):
    if len(a) < 4:
        return []
    d=numpy.diff(a)**2
    if len(matplotlib.pylab.find(d > 0)) < 2:
        return []
    f = scipy.fft(d, 4*len(a))
    f[0] = 0
    # avoid locking onto a harmonic by looking for the fundamental clock frequency
    fundamental_sps = find_fundamental(abs(f))
    # fundamental_sps could be supplied by user instead of by find_fundamental
    p = find_peak(abs(f), len(f)/fundamental_sps)
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + numpy.angle(f[p])/(tau)
    print "approximate fundamental samples per symbol: %f" % (fundamental_sps)
    print "peak frequency index: %d / %d" % (p, len(f))
    print "samples per symbol: %f" % (1.0/cycles_per_sample)
    print "clock cycles per sample: %f" % (cycles_per_sample)
    print "clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase)
    print "clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    print "symbol count: %d" % (len(symbols))
    return symbols
