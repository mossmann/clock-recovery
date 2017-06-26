#!/usr/bin/python2

# recover the chip sequence from a Direct Sequence Spread Spectrum transmission
# see: Reverse Engineering DSSS, Michael Ossmann, REcon 2017

import numpy
import scipy.signal
from matplotlib.pylab import *

tau = numpy.pi * 2
max_samples = 1000000
min_num_chips = 5
max_seq_length = 100000 # in samples
debug = False

# determine the clock frequency
# input: magnitude spectrum of clock signal (numpy array)
# output: FFT bin number of clock frequency
def find_clock_frequency(spectrum):
    maxima = scipy.signal.argrelextrema(spectrum, numpy.greater_equal)[0]
    while maxima[0] < 3:
        maxima = maxima[1:]
    if maxima.any():
        return maxima[matplotlib.pylab.find(spectrum[maxima] > max(spectrum[5:-4])*0.8)[0]]
    else:
        return 0

def midpoint(a):
    return -20.0
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
    d = (numpy.diff(b*1.0) > 0)
    if len(matplotlib.pylab.find(d > 0)) < 2:
        return []
    f = scipy.fft(d, len(a))
    p = find_clock_frequency(abs(f))
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.75 + numpy.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    #if debug:
        #print("peak frequency index: %d / %d" % (p, len(f)))
        #print("samples per symbol: %f" % (1.0/cycles_per_sample))
        #print("clock cycles per sample: %f" % (cycles_per_sample))
        #print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        #print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        #print("symbol count: %d" % (len(symbols)))
    return symbols

# convert soft symbols into bits (assuming binary symbols)
def slice_bits(symbols):
    bits=[]
    for element in symbols:
        if element >= midpoint(symbols):
            bits.append(1)
        else:
            bits.append(0)
    return bits

# input: complex valued samples
# output: signed FFT bin number of detected frequency
def detect_frequency_offset(samples):
    a = array(samples)
    bin = find_clock_frequency(abs(scipy.fft(a*a)))
    if bin > len(a) // 2:
        bin -= len(a)
    return bin // 2

# input: complex valued samples
#        signed FFT bin number of center frequency
# output: frequency shifted samples
def correct_frequency_offset(samples, offset):
    a = array(samples)
    original_fft = scipy.fft(a)
    if offset < 0:
        offset = len(a) + offset
    shifted_fft = append(original_fft[offset:], original_fft[:offset])
    return scipy.ifft(shifted_fft, len(a))

# input: complex valued samples
#        input signal must be centered at 0 frequency
# output: FFT bin number of chip rate
def detect_chip_rate(samples):
    a = array(samples)
    return find_clock_frequency(abs(scipy.fft(a*a)))

# input: complex valued samples, FFT bin number of chip rate
#        input signal must be centered at 0 frequency
# output: number of chips found in repetitive chip sequence
def detect_chip_sequence_length(samples, chip_rate):
    chip_period = int(round(float(len(samples)) / chip_rate))
    shifted = roll(samples, chip_period)
    differential = samples - shifted
    f = scipy.fft(abs(differential))
    power_spectrum = f * conj(f)
    autocorrelation = abs(scipy.ifft(power_spectrum))
    fundamentals = []
    max_num_chips = max_seq_length // chip_period
    for i in range(chip_period*min_num_chips, max_seq_length):
        fundamentals.append(sum((autocorrelation[::i])[:len(samples)//max_num_chips]))
    return (argmax(fundamentals)+chip_period*min_num_chips) // chip_period

# input: complex valued samples, FFT bin number of chip rate, length of chip sequence
#        input signal must be centered at 0 frequency
# output: list of binary chips
def determine_chip_sequence(samples, chip_rate, sequence_length):
    seq_period = int(round(float(len(samples) * sequence_length) / chip_rate))
    shifted = roll(samples, seq_period)
    comparison = real(samples * conj(shifted))
    filtered = convolve(array((1.0/seq_period,)*seq_period), comparison)
    minima = scipy.signal.argrelextrema(filtered, numpy.less)[0]
    while minima[0] < seq_period:
        minima = minima[1:]
    threshold = min(filtered[seq_period:])*0.8
    peaks = []
    for m in minima:
        if filtered[m] < threshold:
            peaks.append(m)
    sum = zeros(seq_period)
    for n in matplotlib.pylab.find(filtered[peaks] < min(filtered[seq_period:])*0.8):
        if len(peaks) > (n + 1):
            if peaks[n+1] < (peaks[n]+seq_period) and filtered[peaks[n+1]] < filtered[peaks[n]]:
                continue
        if n > 0:
            if peaks[n-1] > (peaks[n]-seq_period) and filtered[peaks[n-1]] < filtered[peaks[n]]:
                continue
        sum += comparison[(peaks[n]-seq_period):peaks[n]]
    symbols=wpcr(sum)
    transitions=slice_bits(symbols)
    chips = [1,]
    for t in transitions:
        if t == 1:
            chips.append(chips[-1] ^ 1)
        else:
            chips.append(chips[-1])
    return chips[1:]

def reverse_dsss(samples):
    offset = detect_frequency_offset(samples)
    corrected = correct_frequency_offset(samples, offset)
    chip_rate = detect_chip_rate(corrected)
    sequence_length = detect_chip_sequence_length(corrected, chip_rate)
    sequence = determine_chip_sequence(corrected, chip_rate, sequence_length)
    if debug:
        print("corrected frequency offset: %d / %d" % (offset, len(samples)))
        print("detected chip rate: %d / %d" % (chip_rate, len(samples)))
        print("detected chip sequence length: %d" % (sequence_length))
        print("chip sequence:")
        print(sequence)

# If called directly from command line, take input file (or stdin) as a stream
# of floats and print binary symbols found therein.
if __name__ == '__main__':
    import sys
    import struct
    debug = True
    samples = []
    if len(sys.argv) > 1:
        if sys.argv[1] == '-':
            file = sys.stdin
        else:
            file = open(sys.argv[1])
    else:
        file = sys.stdin

    data=file.read(8 * max_samples)
    floats=struct.unpack('f'*(len(data)/4), data)
    for i in range(0, len(floats), 2):
        samples.append(floats[i] + 1j * floats[i+1])
    s = array(samples)

    reverse_dsss(samples)
    file.close()
