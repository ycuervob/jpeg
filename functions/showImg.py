import numpy as np
import cv2

def createImg(compressionColor,ymatrix, crmatrix, cbmatrix):

    compressionColor = np.zeros((len(ymatrix), len(ymatrix[0]), 3), np.float32)
    for i in range(len(ymatrix)):
        for j in range(len(ymatrix[0])):
            try:
                compressionColor[i, j, 0] = ymatrix[i, j]
                compressionColor[i, j, 1] = crmatrix[i//2, j//2]
                compressionColor[i, j, 2] = cbmatrix[i//2, j//2]
            except IndexError:
                continue

    compressionColorRGB = cv2.cvtColor(
        compressionColor.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return compressionColor