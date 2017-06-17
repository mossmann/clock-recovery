#!/usr/bin/python2

import numpy
import scipy.signal
from matplotlib.pylab import *

tau = numpy.pi * 2
max_samples = 1000000
debug = False

# determine the clock frequency
# input: magnitude spectrum of clock signal (numpy array)
# output: FFT bin number of clock frequency
def find_clock_frequency(spectrum):
    maxima = scipy.signal.argrelextrema(spectrum, numpy.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if maxima.any():
        return maxima[matplotlib.pylab.find(spectrum[maxima] > max(spectrum[2:-1])*0.8)[0]]
    else:
        return 0

def midpoint(a):
    high = []
    low = []
    average = mean(a)
    for i in range(len(a)):
        if a[i] > average:
            high.append(a[i])
        else:
            low.append(a[i])
    return (median(high) + median(low)) / 2

# whole packet clock recovery
# input: real valued NRZ-like waveform (array, tuple, or list)
#        must have at least 2 samples per symbol
#        must have at least 2 symbol transitions
# output: list of symbols
def wpcr(a):
    if len(a) < 4:
        return []
    b = a > midpoint(a)
    d = numpy.diff(b)**2
    if len(matplotlib.pylab.find(d > 0)) < 2:
        return []
    f = scipy.fft(d, len(a))
    p = find_clock_frequency(abs(f))
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + numpy.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols

# convert soft symbols into bits (assuming binary symbols)
def slice_bits(symbols):
    bits=[]
    for element in symbols:
        if element >= numpy.average(symbols):
            bits.append(1)
        else:
            bits.append(0)
    return bits

# If called directly from command line, take input file (or stdin) as a stream
# of floats and print binary symbols found therein.
if __name__ == '__main__':
    import sys
    import struct
    debug = True
    if len(sys.argv) > 1:
        if sys.argv[1] == '-':
            file = sys.stdin
        else:
            file = open(sys.argv[1])
    else:
        file = sys.stdin
    data=file.read(4 * max_samples)
    samples=struct.unpack('f'*(len(data)/4), data)
    symbols=wpcr(samples)
    bits=slice_bits(symbols)
    print(bits)
    file.close()
