# python 3.9.5
from PIL import Image as im
from functions import *
from collections import Counter
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import ceil
import sys
sys.setrecursionlimit(10000)

# define quantization tables
QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # luminance quantization table
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # chrominance quantization table
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])
# define window size
windowSize = len(QTY)

# read image
imgOriginal = imread('//home//yeison//Documents//GitHub//jpeg//migato.png')

# show imge
plt.imshow(imgOriginal)
plt.show()

# convert BGR to YCrCb
img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCrCb)
width = len(img[0])
height = len(img)
y = np.zeros((height, width), np.float32) + img[:, :, 0]
cr = np.zeros((height, width), np.float32) + img[:, :, 1]
cb = np.zeros((height, width), np.float32) + img[:, :, 2]

# show imge
plt.imshow(img)
plt.show()

# size of the image in bits before compression
totalNumberOfBitsWithoutCompression = len(y) * len(y[0]) * 8 + len(cb) * len(cb[0]) * 8 + len(cr) * len(cr[0]) * 8



# 4: 2: 2 subsampling is used
# another subsampling scheme can be used
# thus chrominance channels should be sub-sampled
# define subsampling factors in both horizontal and vertical directions
SSH, SSV = 2, 2

# filter the chrominance channels using a 2x2 averaging filter # another type of filter can be used
crSub = cr[::SSV, ::SSH]
cbSub = cb[::SSV, ::SSH]

# check if padding is needed,
# if yes define empty arrays to pad each channel DCT with zeros if necessary
yWidth, yLength = ceil(len(y[0]) / windowSize) * windowSize, ceil(len(y) / windowSize) * windowSize

if (len(y[0]) % windowSize == 0) and (len(y) % windowSize == 0):
    yPadded = y.copy()
else:
    yPadded = np.zeros((yLength, yWidth))
    for i in range(len(y)):
        for j in range(len(y[0])):
            yPadded[i, j] += y[i, j]

# chrominance channels have the same dimensions, meaning both can be padded in one loop
cWidth, cLength = ceil(len(cbSub[0]) / windowSize) * windowSize, ceil(len(cbSub) / windowSize) * windowSize

if (len(cbSub[0]) % windowSize == 0) and (len(cbSub) % windowSize == 0):
    crPadded = crSub.copy()
    cbPadded = cbSub.copy()
# since chrominance channels have the same dimensions, one loop is enough
else:
    crPadded = np.zeros((cLength, cWidth))
    cbPadded = np.zeros((cLength, cWidth))
    for i in range(len(crSub)):
        for j in range(len(crSub[0])):
            crPadded[i, j] += crSub[i, j]
            cbPadded[i, j] += cbSub[i, j]


imgrec = np.zeros((len(yPadded), len(yPadded[0]), 3), np.float32)
for i in range(len(yPadded)):
    for j in range(len(yPadded[0])):
        try:
            imgrec[i, j, 0] = yPadded[i, j]
            imgrec[i, j, 1] = crPadded[i//2, j//2]
            imgrec[i, j, 2] = cbPadded[i//2, j//2]
        except IndexError:
            continue
     
result = cv2.cvtColor(imgrec.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
plt.imshow(result)
plt.show()


# get DCT of each channel
# define three empty matrices
yDct, crDct, cbDct = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))

# number of iteration on x axis and y axis to calculate the luminance cosine transform values
# number of blocks in the horizontal direction for luminance
hBlocksForY = int(len(yDct[0]) / windowSize)
# number of blocks in the vertical direction for luminance
vBlocksForY = int(len(yDct) / windowSize)
# number of iteration on x axis and y axis to calculate the chrominance channels cosine transforms values
# number of blocks in the horizontal direction for chrominance
hBlocksForC = int(len(crDct[0]) / windowSize)
# number of blocks in the vertical direction for chrominance
vBlocksForC = int(len(crDct) / windowSize)

# define 3 empty matrices to store the quantized values
yq, crq, cbq = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))

# and another 3 for the zigzags
yZigzag = np.zeros(((vBlocksForY * hBlocksForY), windowSize * windowSize))
crZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))
cbZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))

# Calculates the DCT for the component Y of the image
apply_dct(vBlocksForY, hBlocksForY, yPadded, yDct, yq, yZigzag, windowSize, QTY)
# Either crq or cbq can be used to compute the number of blocks
apply_dct(vBlocksForC, hBlocksForC, crPadded, crDct, crq, crZigzag, windowSize, QTC)
apply_dct(vBlocksForC, hBlocksForC, cbPadded, cbDct, cbq, cbZigzag, windowSize, QTC)


yiDct = inv_dct(vBlocksForY, hBlocksForY, yq, windowSize)
criDct = inv_dct(vBlocksForC, hBlocksForC, crq, windowSize)
cbiDct = inv_dct(vBlocksForC, hBlocksForC, cbq, windowSize)

imgrec2 = np.zeros((len(yiDct), len(yiDct[0]), 3), np.float32)
for i in range(len(yiDct)):
    for j in range(len(yiDct[0])):
        try:
            imgrec2[i, j, 0] = yiDct[i, j]
            imgrec2[i, j, 1] = criDct[i//2, j//2]
            imgrec2[i, j, 2] = cbiDct[i//2, j//2]
        except IndexError:
            continue
     
result2 = cv2.cvtColor(imgrec2.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
plt.imshow(result2)
plt.show()



# set type for the zigzag vector
yZigzag = yZigzag.astype(np.int16)
# set type for the zigzag vector
crZigzag = crZigzag.astype(np.int16)
cbZigzag = cbZigzag.astype(np.int16)

# find the run length encoding for each channel
# then get the frequency of each component in order to form a Huffman dictionary
yEncoded = run_length_encoding(yZigzag)
yFrequencyTable = get_freq_dict(yEncoded)
yHuffman = find_huffman(yFrequencyTable)

crEncoded = run_length_encoding(crZigzag)
crFrequencyTable = get_freq_dict(crEncoded)
crHuffman = find_huffman(crFrequencyTable)

cbEncoded = run_length_encoding(cbZigzag)
cbFrequencyTable = get_freq_dict(cbEncoded)
cbHuffman = find_huffman(cbFrequencyTable)

# calculate the number of bits to transmit for each channel
# and write them to an output file
file = open("CompressedImage.asfh", "w")
yBitsToTransmit = str()
for value in yEncoded:
    yBitsToTransmit += yHuffman[value]

crBitsToTransmit = str()
for value in crEncoded:
    crBitsToTransmit += crHuffman[value]

cbBitsToTransmit = str()
for value in cbEncoded:
    cbBitsToTransmit += cbHuffman[value]

if file.writable():
    file.write(yBitsToTransmit + "\n" +
               crBitsToTransmit + "\n" + cbBitsToTransmit)

file.close()

totalNumberOfBitsAfterCompression = len(
    yBitsToTransmit) + len(crBitsToTransmit) + len(cbBitsToTransmit)
print("Compression Ratio is " + str(np.round(totalNumberOfBitsWithoutCompression /
      totalNumberOfBitsAfterCompression, 1)))
