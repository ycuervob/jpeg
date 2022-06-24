# python 3.9.5
import cv2
import numpy as np

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
