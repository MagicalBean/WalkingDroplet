import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

# add the filename as a command line argument
# e.g. python fft_test.py <filename>
filename = sys.argv[1]

# read in image
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# calculate fourier transform
img_fft = np.fft.fft2(image)

# shift the zero frequency component to center
img_fft_shifted = np.fft.fftshift(img_fft)

# calculate the magnitude spectrum of the k-space image for visualizing
magnitude_spectrum = np.log(np.abs(img_fft_shifted) + 1)

# shift zero frequency component back to original position
img_ifft_shifted = np.fft.ifftshift(img_fft_shifted)

# calculate inverse fft
img_inv_fft = np.fft.ifft2(img_ifft_shifted)

# output fft image
cv2.imwrite("output/" + filename.split(".")[0] + "-fft.tiff", (magnitude_spectrum * 255.0 / magnitude_spectrum.max()).astype(np.uint8))
cv2.imwrite("output/" + filename.split(".")[0] + "-inv.tiff", np.abs(img_inv_fft).astype(np.uint8))

# preview output with plt
plt.figure(figsize=(10,4))
plt.subplot(131)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.title("Fourier Transformed")
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(133)
plt.title("Inverse Fourier Transformed")
plt.imshow(np.real(img_inv_fft), cmap='gray')
plt.show()