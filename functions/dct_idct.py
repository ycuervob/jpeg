# python 3.9.5
import cv2
import numpy as np


def inv_dct(vBlocksFor_, hBlocksFor_, _Dct, windowSize):
    _IDct = np.zeros((len(_Dct), len(_Dct[0])), np.float32)
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            # Gets the DCT for each section separated by windowSize spaces

            subMatrix_i = i * windowSize
            subMatrix_j = j * windowSize

            _IDct[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize] = cv2.idct(
                _Dct[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize])
    return _IDct


def apply_dct(vBlocksFor_, hBlocksFor_, _Padded, _Dct, _q, _Zigzag, windowSize):
    # Calculates the DCT for the component Y of the image
    for i in range(vBlocksFor_):
        for j in range(hBlocksFor_):
            # Gets the DCT for each section separated by windowSize spaces

            subMatrix_i = i * windowSize
            subMatrix_j = j * windowSize

            _Dct[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize] = cv2.dct(
                _Padded[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize])

            # Once with the DCT then apply the ceil function to get the cuantized values
            _q[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize] = np.round(
                _Dct[subMatrix_i: subMatrix_i + windowSize, subMatrix_j: subMatrix_j + windowSize])

            # Put the matrix form into a vector
            _Zigzag.append(_q[subMatrix_i: subMatrix_i + windowSize,subMatrix_j: subMatrix_j + windowSize].flatten())
