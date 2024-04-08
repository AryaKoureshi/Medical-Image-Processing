# Medical Image Processing

This repository contains code for processing and analyzing MRI images using Python. The project covers a variety of tasks including image loading, manipulation, noise addition, filtering, and analysis. It also explores advanced techniques such as Fourier transforms, PCA, and morphological operations for image enhancement and analysis.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dependencies](#dependencies)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Getting Started
To get started with this project, clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/mri-image-processing.git
```

## Dependencies
This project requires the following dependencies:
- Python 3.x
- Nibabel
- NumPy
- Matplotlib
- OpenCV
- Scikit-image
- Scikit-learn

You can install these dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage
1. **Image Loading and Processing:** Use the provided functions to load and process MRI images.
2. **Noise Addition:** Add salt and pepper noise to images with a specified block size.
3. **Variance Calculation:** Calculate the variance in a block of an image with arbitrary dimensions.
4. **Local Variance Analysis:** Determine the local variance of each piece of the image and identify the block with the highest variance.
5. **Noise Removal:** Use three arbitrary filters for noise removal and compare the results.
6. **SSIM and MAE Calculation:** Calculate Structural Similarity Index Measurement (SSIM) and Median Absolute Error (MAE) on denoised images.
7. **Brightness Level Transformation:** Use Law-Power and Logarithmic brightness level transformation methods on images.
8. **Image Rotation:** Rotate images using Fourier transform.
9. **Fourier Transform Manipulation:** Calculate the Fourier transform of images, change the phase of the images, and analyze the outputs.
10. **Image Reconstruction:** Reconstruct images using the Wiener filter and evaluate the PSNR criterion.
11. **Morphological Operations:** Use morphological operations to produce masks for image segmentation and contrast enhancement.
12. **PCA Compression:** Use PCA transform to compress images and reconstruct them based on a specified number of eigenvalues.

## Contributing
Contributions to this project are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
