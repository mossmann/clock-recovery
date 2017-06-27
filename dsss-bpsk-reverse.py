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
        threshold = max(spectrum[5:-4])*0.8
        indices_above_threshold = numpy.argwhere(spectrum[maxima] > threshold)
        return maxima[indices_above_threshold[0][0]]
    else:
        return 0

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

# input: complex valued samples
#        input signal must be centered at 0 frequency
# output: subset of samples with optimal sampling time for each chip
def extract_chip_samples(samples):
    a = array(samples)
    f = scipy.fft(a*a)
    p = find_clock_frequency(abs(f))
    if 0 == p:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.25 + numpy.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    chip_samples = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            chip_samples.append(a[i])
        clock_phase += cycles_per_sample
    return chip_samples

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
def determine_chip_sequence(samples, sequence_length):
    shifted = roll(samples, sequence_length)
    comparison = real(samples * conj(shifted))
    filtered = convolve(array((1.0/sequence_length,)*sequence_length), comparison)
    threshold = min(filtered) / 2.0
    indices_over_threshold = filtered > threshold
    filtered[indices_over_threshold] = 0
    sums = []
    for i in range(sequence_length):
        sums.append(numpy.sum(filtered[i::sequence_length]))
    start = argmin(sums)
    shifted = roll(samples, 1)
    comparison = real(samples * conj(shifted))
    sums = []
    for i in range(sequence_length):
        sums.append(numpy.sum(comparison[i::sequence_length]))
    transitions = roll(sums, sequence_length - start - 1)
    chips = [0,]
    for t in transitions:
        if t > 0:
            chips.append(chips[-1])
        else:
            chips.append(chips[-1] ^ 1)
    return chips[1:]

def reverse_dsss(samples):
    offset = detect_frequency_offset(samples)
    corrected = correct_frequency_offset(samples, offset)
    chip_samples = extract_chip_samples(corrected)
    sequence_length = detect_chip_sequence_length(corrected, len(chip_samples))
    sequence = determine_chip_sequence(chip_samples, sequence_length)
    if debug:
        print("corrected frequency offset: %d / %d" % (offset, len(samples)))
        print("detected chip rate: %d / %d" % (len(chip_samples), len(samples)))
        print("detected chip sequence length: %d" % (sequence_length))
        print("chip sequence:")
        print(sequence)

def read_from_stdin():
    return numpy.frombuffer(sys.stdin.buffer.read(), dtype=numpy.complex64, count=max_samples)

# If called directly from command line, take input file (or stdin) as a stream
# of floats and print binary symbols found therein.
if __name__ == '__main__':
    import sys
    debug = True
    if len(sys.argv) > 1 and sys.argv[1] != '-':
        samples = numpy.fromfile(sys.argv[1], dtype=numpy.complex64, count=max_samples)
    else:
        samples = read_from_stdin()
    reverse_dsss(samples)
