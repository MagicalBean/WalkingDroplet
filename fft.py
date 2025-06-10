import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def process_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not open or find the image {image_path}")
        sys.exit(1)

    # Compute the Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Compute the Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift)
    img_reconstructed = np.fft.ifft2(f_ishift)
    img_reconstructed = np.abs(img_reconstructed)

    # Save images as TIFF files
    # cv2.imwrite("original.tiff", img)
    cv2.imwrite("output/fourier_transformed.tiff", (magnitude_spectrum * 255 / magnitude_spectrum.max()).astype(np.uint8))
    cv2.imwrite("output/inverse_fourier_transformed.tiff", img_reconstructed.astype(np.uint8))

    # Display the images
    plt.figure(figsize=(10,5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Fourier Transformed")
    plt.imshow(magnitude_spectrum, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Inverse Fourier Transformed")
    plt.imshow(img_reconstructed, cmap='gray')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    process_image(sys.argv[1])
