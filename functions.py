# python 3.9.5
from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from collections import Counter
from scipy.fftpack import fft, dct,idct


def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            break
    return output


def trim(array: np.ndarray) -> np.ndarray:
    """
    in case the trim_zeros function returns an empty array, add a zero to the array to use as the DC component
    :param numpy.ndarray array: array to be trimmed
    :return numpy.ndarray:
    """
    trimmed = np.trim_zeros(array, 'b')
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed


def run_length_encoding(array: np.ndarray) -> list:
    """
    finds the intermediary stream representing the zigzags
    format for DC components is <size><amplitude>
    format for AC components is <run_length, size> <Amplitude of non-zero>
    :param numpy.ndarray array: zigzag vectors in array
    :returns: run length encoded values as an array of tuples
    """
    encoded = list()
    run_length = 0
    eob = ("EOB",)

    for i in range(len(array)):
        for j in range(len(array[i])):
            trimmed = trim(array[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # for the first DC component
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # to compute the difference between DC components
                diff = int(array[i][j] - array[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                run_length = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                run_length += 1
            else:  # intermediary steam representation of the AC components
                encoded.append((run_length, int(trimmed[j]).bit_length(), trimmed[j]))
                run_length = 0
            # send EOB
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded

def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    p_copy = {}
    p_copy2 = {}
    for i in p.keys():
        p_copy[i] = ""
        p_copy2[str(i)] = ""

    while len(p_copy2)>=2:
        a1, a2 = lowest_prob_pair(p_copy2)
        p1, p2 = p_copy2.pop(a1), p_copy2.pop(a2)
        
        for i in a1.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(map(int, i.replace('(','').replace(')','').split(', ')))] += "1"
            else:
                p_copy[('EOB',)] += "1"

        for i in a2.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(map(int, i.replace('(','').replace(')','').split(', ')))] += "0"
            else:
                p_copy[('EOB',)] += "0"

        p_copy2[a1+ "|"+ a2] = p1 + p2

    return p_copy


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]

def inv_dct(vBlocksFor_,hBlocksFor_,_Dct,windowSize):
    _IDct = np.zeros((len(_Dct),len(_Dct[0])), np.float32) 
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            #Gets the DCT for each section separated by windowSize spaces
            _IDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.idct(
                _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
    return _IDct

def apply_dct(vBlocksFor_,hBlocksFor_,_Padded,_Dct,_q,_Zigzag,windowSize):
    #Calculates the DCT for the component Y of the image
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            #Gets the DCT for each section separated by windowSize spaces
            _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
                _Padded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

            #Once with the DCT then apply the ceil function to get the cuantized values
            _q[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.round(
                _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

            #Put the matrix form into a vector
            _Zigzag[i * j] += zigzag(_q[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
