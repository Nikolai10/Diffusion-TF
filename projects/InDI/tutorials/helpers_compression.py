# Copyright 2024 Nikolai Körber. All Rights Reserved.
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
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
import numpy as np
import glob
import os
import re


def export_image(trajectory, gen_path, idx=0):
    """export single image at index idx to png"""

    for t, traj in enumerate(trajectory):
        filename = gen_path + "x" + str(t) + ".png"
        traj = (
            tf.clip_by_value(traj * 127.5 + 127.5, 0.0, 255.0)
                .numpy()
                .astype(np.uint8)
        )

        # Save the tensor as a PNG image
        save_img(filename, traj[idx])  # Convert the tensor to NumPy array for saving


def make_gif(gen_path, fps=3):
    """Use imageio to create an animated gif using the images saved during training"""

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    anim_file = os.path.join(gen_path, 'compression.gif')

    with imageio.get_writer(anim_file, mode='I', duration=0.2, loop=0, fps=fps) as writer:
        filenames = glob.glob(os.path.join(gen_path, 'x*.png'))
        filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f'GIF created at: {anim_file}')
