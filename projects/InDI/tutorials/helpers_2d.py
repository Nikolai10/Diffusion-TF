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

import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
import os
import re


def generate_samples_from_multi_modal_dist(n_data):
    """See https://arxiv.org/pdf/2303.11435.pdf, Fig. 1a (l)"""
    # Number of modes
    num_modes = 4
    # Dimensions
    num_dimensions = 2
    # Mode locations
    modes = np.array([[1, -1], [-1, 1], [1, 1], [-1, -1]])
    # Weights for each mode
    weights = np.ones(num_modes) / num_modes
    # Variance for noise
    variance = 1
    # Generate noisy samples
    samples = []
    labels = []
    for _ in range(n_data):
        # Sample from the multimodal distribution
        selected_mode = modes[np.random.choice(num_modes, p=weights)]
        # Generate noise samples from a multivariate normal distribution
        noise = np.random.multivariate_normal(mean=np.zeros(num_dimensions), cov=variance * np.identity(num_dimensions))
        # Combine mode and noise
        sample = selected_mode + noise
        # append to array
        samples.append(sample)
        labels.append(selected_mode)
    # convert to numpy array
    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return samples, labels, modes


def generate_rotated_circle_points(radius, num_points=20, rotation_angle=0.1):
    """See https://arxiv.org/pdf/2303.11435.pdf, Fig. 1a (r)"""
    # Generate points on a slightly rotated circle
    theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Introduce a small rotation angle
    rotated_theta_values = theta_values + rotation_angle

    x_values = radius * np.cos(rotated_theta_values)
    y_values = radius * np.sin(rotated_theta_values)

    return np.column_stack((x_values, y_values)).astype(np.float32)


def plot_images(trajectory, modes, points_2d, x_hat_zero, regression, gen_path, figsize=(8, 6)):
    """Utility to plot images using the recorded trajectory."""

    for t, traj in enumerate(trajectory):

        # adjust filename
        filename = gen_path + "x" + str(t) + ".png"

        # Create a figure with a specific size
        plt.figure(figsize=figsize)
        plt.grid(True, zorder=0)
        plt.gca().set_axisbelow(True)

        # show modes
        plt.scatter(modes[:, 0], modes[:, 1], marker='o', color="orange", linewidth=1.5, s=100, label='$p(x)$',
                    zorder=3)

        # show test samples
        plt.scatter(points_2d[:, 0], points_2d[:, 1], marker='o', color="blue", s=100, label='$\hat{x}_1$')

        # add xt to legend
        tray = trajectory[0]
        plt.scatter(tray[:, 0], tray[:, 1], marker='o', color="green", linewidth=1.5, label='$\hat{x}_t$', s=100,
                    zorder=0)

        # add xhatt to legend
        plt.scatter([100], [100], marker='o', color="white", edgecolor='black', linewidth=1.5,
                    s=100, label='$\hat{x}_0$',
                    zorder=1)

        # show xt
        for idx, tt in enumerate(range(t)):
            tray = trajectory[idx + 1]
            plt.scatter(tray[:, 0], tray[:, 1], marker='o', color="green", s=100 * (tt + 1) / (len(trajectory) - 1),
                        zorder=-1)

        # highlight final reconstruction
        if t == len(trajectory) - 1:
            plt.scatter(x_hat_zero[:, 0], x_hat_zero[:, 1], marker='o', color="white", edgecolor='black', linewidth=1.5,
                        s=100, zorder=2)

        # show regression results
        plt.scatter(regression[:, 0], regression[:, 1], marker='o', color="red", edgecolor="orangered", linewidth=1.5,
                    label='regression',
                    s=100, zorder=0)

        # Connect points with dashed lines
        for i in range(len(points_2d)):
            plt.plot([points_2d[i, 0], regression[i, 0]], [points_2d[i, 1], regression[i, 1]], linestyle='dashed',
                     color='red', zorder=0)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.75), fontsize=16)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def make_gif(gen_path):
    """Use imageio to create an animated gif using the images saved during training"""

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    anim_file = os.path.join(gen_path, '2d.gif')

    with imageio.get_writer(anim_file, mode='I', duration=0.2, loop=0, fps=2) as writer:
        filenames = glob.glob(os.path.join(gen_path, 'x*.png'))
        filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f'GIF created at: {anim_file}')
