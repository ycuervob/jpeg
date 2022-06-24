# python 3.9.5
from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from collections import Counter
from scipy.fftpack import fft, dct, idct


def zigzag(matrix: np.ndarray) -> np.ndarray:
    return matrix.flatten()


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

def run_length_encod(seq):
  compressed = []
  count = 1
  char = seq[0]
  print(char)
  for i in range(1,len(seq)):
    if seq[i] == char:
      count = count + 1
    else :
      compressed.append((char,count))
      char = seq[i]
      count = 1
  compressed.append((char,count))
  return compressed
 
def run_length_decoding(compressed_seq):
  seq = []
  for i in range(0,len(compressed_seq)):
    for j in range(compressed_seq[i][1]):
        seq.append(int(compressed_seq[i][0]))
 
  return(seq)

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


def decode_huffman(cod: str, invHuffman: dict) -> list:
    print("decoding huffman...")
    decodingy = []
    index_init = 0
    for index_fin in range(len(cod)):
        if cod[index_init:index_fin] in invHuffman.keys():
            decodingy.append(invHuffman[cod[index_init:index_fin]])
            index_init = index_fin
    print("complete")
    return decodingy


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

    while len(p_copy2) >= 2:
        a1, a2 = lowest_prob_pair(p_copy2)
        p1, p2 = p_copy2.pop(a1), p_copy2.pop(a2)

        for i in a1.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(
                    map(int, i.replace('(', '').replace(')', '').split(', ')))] += "1"
            else:
                p_copy[('EOB',)] += "1"

        for i in a2.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(
                    map(int, i.replace('(', '').replace(')', '').split(', ')))] += "0"
            else:
                p_copy[('EOB',)] += "0"

        p_copy2[a1 + "|" + a2] = p1 + p2

    for i in p_copy.keys():
        p_copy[i] = p_copy[i][::-1]
    return p_copy


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]


def inv_dct(vBlocksFor_, hBlocksFor_, _Dct, windowSize):
    _IDct = np.zeros((len(_Dct), len(_Dct[0])), np.float32)
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            # Gets the DCT for each section separated by windowSize spaces
            _IDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.idct(
                _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
    return _IDct


def apply_dct(vBlocksFor_, hBlocksFor_, _Padded, _Dct, _q, _Zigzag, windowSize):
    # Calculates the DCT for the component Y of the image
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            # Gets the DCT for each section separated by windowSize spaces
            _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
                _Padded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

            # Once with the DCT then apply the ceil function to get the cuantized values
            _q[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.round(
                _Dct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

            # Put the matrix form into a vector
            _Zigzag.append(_q[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize].flatten())
