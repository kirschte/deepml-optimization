#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:49:56 2019

@author: Moritz Kirschte
"""

import sys

import numpy as np
from PIL import Image


def make_pca(X, cov, mean):
    """Reduces RGB-Image to greyscale with PCA.

    # Arguments
        X: np.array[3, width*height]. The input image.
        cov: np.array[3, 3]. Covariance matrix of the input image.
        mean: np.array[3, 1]. Means per channel of the input image.

    # Returns
        Input image to one channel reduced. Similar to a real greyscale.
    """

    eigenval, eigenvec = np.linalg.eig(cov)
    print('Eigenvalue significance (per channel):', eigenval)

    # standardize
    X = X - mean
    stdev = np.sqrt(np.diag(cov))
    for i in stdev:
        if i == 0:
            raise NameError('Standarddeviation is zero')
    X = X / stdev.reshape((-1, 1))

    # pca
    vec_2d = eigenvec[0:1]
    z = np.dot(vec_2d, X)
    diff = np.max(z) - np.min(z)
    z_new = z * 255/diff
    z_new = z_new - np.min(z_new)
    return -z_new


def image_to_pca(img):
    """Wrapper for `make_pca` to pre- and post-process an image.

    # Arguments:
        img: PIL.Image. RGB image.

    # Returns:
        PIL.Image. Greyscaled image.
    """

    arr = np.array(img)
    flatten = arr.T.reshape((-1, arr.shape[0] * arr.shape[1]))
    z = make_pca(flatten,
                 np.cov(flatten),
                 np.mean(flatten, axis=1, keepdims=True))
    return Image.fromarray(np.uint8(z.reshape(512, 512).T), mode='L')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Input/Output name needs to be specified.')
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # ------------------------
    img_in = Image.open(input_file)
    img_out = image_to_pca(img_in)
    # ------------------------
    img_out.save(output_file)
