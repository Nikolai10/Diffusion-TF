# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tchambon/posts/blob/main/iadb-2D/IADB_2d.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os
import re


def load_img(filename):
    """load PNG image"""
    image = imageio.imread(filename).astype("float32")[:, :, 0:3] / 255.0
    image = image[np.newaxis, ...]
    return image


def generate_samples_from_img(p, n_data):
    """generate Ndata from PDF p represented as an image using rejection sampling"""
    maxPDFvalue = np.max(p)
    samples = np.zeros((n_data, 2), dtype=np.float32)
    for n in tqdm(range(n_data), "generateSamplesFromImage"):
        while True:
            # random location in [0, 1]²
            x = np.random.rand()
            y = np.random.rand()
            # discrete pixel coordinates of (x,y)
            i = int(x * p.shape[1])
            j = int(y * p.shape[2])
            # random value
            u = np.random.rand()
            # keep or reject?
            if p[0, i, j, 0] / maxPDFvalue >= u:
                samples[n, 0] = x
                samples[n, 1] = y
                break
    return samples


def export(x, filename):
    """create plot with samples"""
    plt.figure()  # Create a new figure
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_aspect('equal', adjustable='box')
    for i in range(x.shape[0]):
        plt.plot(x[i, 1], 1 - x[i, 0], 'bo', markersize=1)
    plt.savefig(filename)
    plt.close()


def show_imgs(img_a, img_b, name_a, name_b, figsize=(6, 6)):
    """Create a subplot with 2 columns and 1 row"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Display the first image
    axes[0].imshow(img_a)
    axes[0].set_title(name_a)
    axes[0].axis('off')

    # Display the second image
    axes[1].imshow(img_b)
    axes[1].set_title(name_b)
    axes[1].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def make_gif(gen_path):
    """Use imageio to create an animated gif using the images saved during training"""

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    anim_file = os.path.join(gen_path, '2d.gif')

    with imageio.get_writer(anim_file, mode='I', duration=0.2, loop=0) as writer:
        filenames = glob.glob(os.path.join(gen_path, 'x*.png'))
        filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f'GIF created at: {anim_file}')
