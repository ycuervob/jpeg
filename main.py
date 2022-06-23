# python 3.9.5
from PIL import Image as im
from cv2 import line
from functions import *
from collections import Counter
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import ceil

# define window size
windowSize = 8

# read image
imgOriginal = imread('C:\\Users\\Usuario\\Desktop\\JPEG compresor\\migato.png')


# convert BGR to YCrCb
imgInYCrCb = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCrCb)
width = len(imgInYCrCb[0])
height = len(imgInYCrCb)
y = np.zeros((height, width), np.float32) + imgInYCrCb[:, :, 0]
cr = np.zeros((height, width), np.float32) + imgInYCrCb[:, :, 1]
cb = np.zeros((height, width), np.float32) + imgInYCrCb[:, :, 2]

# size of the image in bits before compression
totalNumberOfBitsWithoutCompression = len(
    y) * len(y[0]) * 8 + len(cb) * len(cb[0]) * 8 + len(cr) * len(cr[0]) * 8

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
yWidth, yLength = ceil(len(y[0]) / windowSize) * \
    windowSize, ceil(len(y) / windowSize) * windowSize

if (len(y[0]) % windowSize == 0) and (len(y) % windowSize == 0):
    yPadded = y.copy()
else:
    yPadded = np.zeros((yLength, yWidth))
    for i in range(len(y)):
        for j in range(len(y[0])):
            yPadded[i, j] += y[i, j]

# chrominance channels have the same dimensions, meaning both can be padded in one loop
cWidth, cLength = ceil(len(cbSub[0]) / windowSize) * \
    windowSize, ceil(len(cbSub) / windowSize) * windowSize

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


compressionColor = np.zeros((len(yPadded), len(yPadded[0]), 3), np.float32)
for i in range(len(yPadded)):
    for j in range(len(yPadded[0])):
        try:
            compressionColor[i, j, 0] = yPadded[i, j]
            compressionColor[i, j, 1] = crPadded[i//2, j//2]
            compressionColor[i, j, 2] = cbPadded[i//2, j//2]
        except IndexError:
            continue

compressionColorRGB = cv2.cvtColor(
    compressionColor.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


# get DCT of each channel
# define three empty matrices
yDct, crDct, cbDct = np.zeros((yLength, yWidth)), np.zeros(
    (cLength, cWidth)), np.zeros((cLength, cWidth))

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
yq, crq, cbq = np.zeros((yLength, yWidth)), np.zeros(
    (cLength, cWidth)), np.zeros((cLength, cWidth))

# and another 3 for the zigzags
yZigzag = np.zeros(((vBlocksForY * hBlocksForY), windowSize * windowSize))
crZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))
cbZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))

# Calculates the DCT for the component Y of the image
apply_dct(vBlocksForY, hBlocksForY, yPadded, yDct, yq, yZigzag, windowSize)
# Either crq or cbq can be used to compute the number of blocks
apply_dct(vBlocksForC, hBlocksForC, crPadded, crDct, crq, crZigzag, windowSize)
apply_dct(vBlocksForC, hBlocksForC, cbPadded, cbDct, cbq, cbZigzag, windowSize)


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

yBitsToTransmit = []
for value in yEncoded:
    yBitsToTransmit.append(yHuffman[value])

crBitsToTransmit = []
for value in crEncoded:
    crBitsToTransmit.append(crHuffman[value])

cbBitsToTransmit = []
for value in cbEncoded:
    cbBitsToTransmit.append(cbHuffman[value])


# show imge
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(imgOriginal)
plt.axis('off')
plt.title('imgOriginal')

plt.subplot(1, 3, 2)
plt.imshow(imgInYCrCb)
plt.axis('off')
plt.title('imgInYCrCb')

plt.subplot(1, 3, 3)
plt.imshow(compressionColorRGB)
plt.axis('off')
plt.title('compressionColorRGB')

plt.show()

yrealBits = "".join(yBitsToTransmit)
crrealBits = "".join(crBitsToTransmit)
cbrealBits = "".join(cbBitsToTransmit)

file = open("CompressedImage.asfh", "w")
if file.writable():
    file.write(yrealBits + "\n" + crrealBits + "\n" + cbrealBits)
file.close()

totalNumberOfBitsAfterCompression = len(yrealBits) + len(crrealBits) + len(cbrealBits)
print("Compression Ratio is " + str(np.round(totalNumberOfBitsWithoutCompression/totalNumberOfBitsAfterCompression, 1)))


#Descompresion -----------------

invyHuffman = {}
for i in yHuffman.keys():
    invyHuffman[yHuffman[i]] = i

invcrHuffman = {}
for i in crHuffman.keys():
    invcrHuffman[crHuffman[i]] = i

invcbHuffman = {}
for i in cbHuffman.keys():
    invcbHuffman[cbHuffman[i]] = i

decodeFile = open("CompressedImage.asfh", "r")
yread = decodeFile.readline()
crread = decodeFile.readline()
cbread = decodeFile.readline()
decodeFile.close()

ydecoded = decode_huffman(yread,invyHuffman)
print(ydecoded[0:20])
# crdecoded = decode_huffman(crread,invcrHuffman)
# cbdecoded = decode_huffman(cbread,invcbHuffman)






