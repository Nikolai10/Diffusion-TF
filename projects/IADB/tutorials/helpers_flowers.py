# Copyright 2023 Nikolai Körber. All Rights Reserved.
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
import tensorflow as tf
import numpy as np
import glob
import os
import re


def plot_images(trajectory, gen_path, num_rows=4, num_cols=8, figsize=(12, 5)):
    """Utility to plot images using the recorded trajectory."""

    for t, traj in enumerate(trajectory):

        filename = gen_path + "x" + str(t) + ".png"
        generated_samples = (
            tf.clip_by_value(traj * 127.5 + 127.5, 0.0, 255.0)
                .numpy()
                .astype(np.uint8)
        )

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def make_gif(gen_path):
    """
    Use imageio to create an animated gif using the images saved during training; based on
    https://www.tensorflow.org/tutorials/generative/dcgan

    :param gen_path: Path to the directory containing images
    :return: None
    """
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)

    anim_file = os.path.join(gen_path, 'flowers.gif')
    filenames = glob.glob(os.path.join(gen_path, 'x*.png'))
    filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))

    if not filenames:
        print("No image files found to create GIF.")
        return

    filenames = filenames
    last = -1

    with imageio.get_writer(anim_file, mode='I', loop=0) as writer:
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
