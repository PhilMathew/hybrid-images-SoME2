import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np



def gaussian(i, j, sigma):
    """
    Computes the Gaussian of (i, j) with the given value of sigma
    """
    return np.exp(-(i**2 + j**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def gauss_kernel(n, sigma):
    """
    Creates a square gaussian kernel

    :param n: size of kernel
    NOTE: if n is not odd, then it will be incremented by 1
    :param sigma: variance of kernel
    :return: Gaussian kernel of size s with variance sigma
    """
    if n % 2 == 0:
        n += 1
        
    vals = np.linspace(-n / 2, n / 2, num=n)
    xi, yi = np.meshgrid(vals, vals)
    h = gaussian(xi, yi, sigma)

    return h


def filter_convolve(img, kernel):
    """
    Convolves the given image with the given convolution kernel
    NOTE: Image is padded at edges during convolution so that the filter can be applied to boundary pixels

    :param img: image to apply filter to
    :param kernel: kernel to convolve the image with (assumed to be square)
    :return: convolved image
    """
    pad_width = kernel.shape[0] // 2 # assuming square kernel
    conv_input = np.pad(img, (pad_width, pad_width), mode='constant', constant_values=0) # pads image so that we can run the kernel on the edges
    
    # Sliding window convolution with stride of 1
    ret_img = np.zeros_like(img)
    for i in range(ret_img.shape[0]):
        for j in range(ret_img.shape[1]):
            img_slice = conv_input[i:(i + (2 * pad_width) + 1), j:(j + (2 * pad_width) + 1)] # grab a part of the image centered on the current pixel
            elem_prod = img_slice * kernel # take element-wise product
            ret_img[i][j] = np.sum(elem_prod) # assign sum of entries in above product to the output pixel at (i, j)
    
    return ret_img


def hybridize(img1, img2, sigma1=5, sigma2=5):
    img1 = img1 / 255
    img2 = img2 / 255
    
    # Expecting NHWC format
    n_channels = img1.shape[-1]
    
    # Create blurring kernels
    k1 = gauss_kernel(3 * sigma1, sigma1)
    k2 = gauss_kernel(3 * sigma2, sigma2)
    
    # Blur both images across all channels
    blur_img1, blur_img2, = np.zeros_like(img1), np.zeros_like(img2)
    for i in range(n_channels):
        blur_img1[:, :, i] = filter_convolve(img1[:, :, i], k1)
        blur_img2[:, :, i] = filter_convolve(img2[:, :, i], k2)
    
    # Subtract blurred image to get highpass-filtered version of image 2
    highpass_img2 = img2 - blur_img2
    
    # Add lowpassed version of image 1 to highpassed version of image 2
    hybrid = blur_img1 + highpass_img2
    
    return hybrid * 255


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = ArgumentParser(description='Hybrid Image Creator')
    parser.add_argument('--img1', dest='img1_path', type=str, help='Path to irst image to be used in creating the hybrid image')
    parser.add_argument('--img2', dest='img2_path', type=str, help='Path to second image to be used in creating the hybrid image')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='.', help='Directory to output hybrid image to')        
    args = parser.parse_args()
    
    # Make output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    
    # Load images
    img1_path, img2_path = Path(args.img1_path), Path(args.img2_path)
    img1, img2 = cv2.imread(str(img1_path)), cv2.imread(str(img2_path))
    
    # Hybridize images
    hybrid = hybridize(img1, img2, sigma1=5, sigma2=9)
    
    # Save out hybrid image and its smaller coutnerpart
    cv2.imwrite(str(output_dir / 'hybrid_fullsize.png'), hybrid)
    small_size = [i // 8 for i in hybrid.shape[:2]]
    cv2.imwrite(str(output_dir / 'hybrid_small.png'), cv2.resize(hybrid, small_size))


if __name__ == '__main__':
    main()
