import numpy as np
import matplotlib.pyplot as plt
import cv2

pepperIm = cv2.imread("peppers.bmp")
lenaIm = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)

def FindInfo():
    height, width = lenaIm.shape
    maxIntensity, minIntensity, sumIntensity, medianIntensity = 0, 255, 0, 0
    intensityLength = height * width
    histogram = [0] * 256

    # obtain the max, min, and mean intensities
    i = 0
    while i < height:
        j = 0
        while j < width:
            intensityValue = lenaIm[i][j]
            sumIntensity += intensityValue
            histogram[intensityValue] += 1

            if intensityValue > maxIntensity:
                maxIntensity = intensityValue

            if intensityValue < minIntensity:
                minIntensity = intensityValue
            j += 1
        i += 1

    meanIntensity = sumIntensity / intensityLength

    # find the median using the histogram
    cumulative, intensity = 0, 0
    while intensity < 256:
        cumulative += histogram[intensity]
        if cumulative >= (intensityLength + 1) // 2:
            medianIntensity = intensity
            break
        intensity += 1
    
    return maxIntensity, minIntensity, meanIntensity, medianIntensity


def GenerateBlurImage(image, blockSize):
    imageDimension = image.shape
    height, width = imageDimension[0], imageDimension[1]
    blurredImage = np.copy(image)

    for i in range(0, height, blockSize):
        for j in range(0, width, blockSize):
            block = image[i:i+blockSize, j:j+blockSize]

            if len(block.shape) == 3:
                average_intensity = np.mean(block, axis=(0, 1))
            else:
                average_intensity = np.mean(block)

            blurredImage[i:i+blockSize, j:j+blockSize] = average_intensity

    return blurredImage

if pepperIm is not None and lenaIm is not None:

    # QUESTION 1
    pepperImRBG = cv2.cvtColor(pepperIm, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.suptitle('Original Images')

    plt.subplot(1, 2, 1)
    plt.imshow(pepperImRBG)

    plt.subplot(1, 2, 2)
    plt.imshow(lenaIm, cmap='gray')

    # QUESTION 2
    pepperGrayIm = cv2.cvtColor(pepperIm, cv2.COLOR_BGR2GRAY)
    pepperGrayImT = cv2.transpose(pepperGrayIm)
    pepperGrayImF = np.flip(pepperGrayIm, axis=1)
    pepperGrayImH = np.flip(pepperGrayIm, axis=0)

    plt.figure(figsize=(12, 12))
    plt.suptitle('Gray Pepper Images')

    plt.subplot(2, 2, 1)
    plt.imshow(pepperGrayIm, cmap='gray')
    plt.title("pepperGrayIm")

    plt.subplot(2, 2, 2)
    plt.imshow(pepperGrayImT, cmap='gray')
    plt.title("pepperGrayImT")

    plt.subplot(2, 2, 3)
    plt.imshow(pepperGrayImF, cmap='gray')
    plt.title("pepperGrayImF")

    plt.subplot(2, 2, 4)
    plt.imshow(pepperGrayImH, cmap='gray')
    plt.title("pepperGrayImH")

    # QUESTION 3: mean, median, minimum, and maximum
    builtinMaximum = np.max(lenaIm)
    builtinMinimum = np.min(lenaIm)
    builtinMean = np.mean(lenaIm)
    builtinMedian = np.median(lenaIm)

    maximum, minimum, mean, median = FindInfo()

    if builtinMaximum == maximum:
        print("maximums match: ", builtinMaximum)
    if builtinMinimum == minimum:
        print("minimums match: ", builtinMinimum)
    if builtinMean == mean:
        print("means match: ", builtinMean)
    if builtinMedian == median:
        print("medians match: ", builtinMedian)

    # QUESTION 4: Normalize "lenaIm"
    normalizedLenaIm = lenaIm.astype(float) / 255.0
    
    plt.figure(figsize=(6, 6))
    plt.imshow(normalizedLenaIm, cmap='gray')
    plt.title("Normalized Grayscale Image")

    height, width = normalizedLenaIm.shape
    quarterHeight = height // 4

    firstQuarter = normalizedLenaIm[:quarterHeight, :]
    secondQuarter = normalizedLenaIm[quarterHeight:2 * quarterHeight, :]
    thirdQuarter = normalizedLenaIm[2 * quarterHeight:3 * quarterHeight, :]
    fourthQuarter = normalizedLenaIm[3 * quarterHeight:, :]

    processedSecondQuarter = secondQuarter ** 1.25
    processedFourthQuarter = fourthQuarter ** 0.25
    processedNormalizedLenaIm = np.vstack((firstQuarter, processedSecondQuarter, thirdQuarter, processedFourthQuarter))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(processedNormalizedLenaIm, cmap='gray')
    plt.title("Processed Grayscale Image")

    file_name = "TylerJohnston_processedNormalizedLenaIm.jpg"
    cv2.imwrite(file_name, (processedNormalizedLenaIm * 255).astype(np.uint8))

    # QUESTION 5: 
    pepperGrayImN = pepperGrayIm.astype(float) / 255.0
    threshold = 0.37

    bw1 = (pepperGrayImN > threshold).astype(int)
    bw2 = [[1 if pixel > threshold else 0 for pixel in row] for row in pepperGrayImN]
    _, bw3 = cv2.threshold(pepperGrayImN, threshold, 1, cv2.THRESH_BINARY)

    if np.array_equal(bw1, bw2) and np.array_equal(bw1, bw3) and np.array_equal(bw2, bw3):
        print("Both my methods worked")
    elif not np.array_equal(bw1, bw3) and np.array_equal(bw2, bw3):
        print("Method 1 worked but Method 2 did not work")
    elif np.array_equal(bw1, bw3) and not np.array_equal(bw2, bw3):
        print("Method 2 worked but Method 1 did not work")
    else:
        print("None of my methods worked")

    plt.figure(figsize=(12, 4))
    plt.suptitle("Binary Thresholding")

    plt.subplot(1, 3, 1)
    plt.imshow(bw1, cmap='gray')
    plt.title("First Method")

    plt.subplot(1, 3, 2)
    plt.imshow(bw2, cmap='gray')
    plt.title("Second Method")

    plt.subplot(1, 3, 3)
    plt.imshow(bw3, cmap='gray')
    plt.title("Third Method")

    # QUESTION 6:

    pepperImBlur = GenerateBlurImage(pepperIm, 4)
    lenaImBlur = GenerateBlurImage(lenaIm, 8)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(pepperIm, cv2.COLOR_BGR2RGB))
    plt.title("Original Color Image (pepperIm)")

    plt.subplot(2, 2, 2)
    plt.imshow(lenaIm, cmap='gray')
    plt.title("Original Grayscale Image (lenaIm)")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(pepperImBlur, cv2.COLOR_BGR2RGB))
    plt.title("Blurred Color Image (pepperImBlur)")

    plt.subplot(2, 2, 4)
    plt.imshow(lenaImBlur, cmap='gray')
    plt.title("Blurred Grayscale Image (lenaImBlur)")

    plt.show()
else:
    print("peppers.bmp and Lena.jpg are either not downloaded or are not in this directory")
