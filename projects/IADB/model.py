# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# - https://github.com/tchambon/IADB
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
"""
This is a reimplementation of IADB published in:
E. Heitz and L. Belcour and T. Chambon:
"Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model"
SIGGRAPH, 2023
https://arxiv.org/abs/2305.03486
"""
import tensorflow as tf


class IADBModel(tf.keras.Model):
    def __init__(self, network, rank=4):
        super().__init__()
        self.network = network
        self.rank = rank  # support various data formats

    def broadcast_alpha(self, alpha):
        """broadcast to rank"""
        if self.rank == 2:  # e.g. 2d points
            return tf.expand_dims(alpha, axis=-1)
        elif self.rank == 4:  # e.g. RGB images
            return tf.expand_dims(tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1), axis=-1)
        else:
            raise NotImplementedError("Rank {} not yet supported".format(self.rank))

    def train_step(self, samples):
        """https://arxiv.org/abs/2305.03486, algorithm 3"""

        # 1. Prepare x0, x1
        if isinstance(samples, tuple) and len(samples) == 2:
            x0, x1 = samples
        else:
            x1 = samples
            x0 = tf.random.normal(shape=tf.shape(x1))

        # 2. Get the batch size
        bs = tf.shape(x0)[0]

        # 3. Sample alpha uniformly
        alpha = tf.random.uniform((bs,), dtype=tf.float32)
        alpha_bc = self.broadcast_alpha(alpha)

        with tf.GradientTape() as tape:
            # 4. Obtain x_alpha
            x_alpha = alpha_bc * x1 + (1 - alpha_bc) * x0

            # 5. Pass the interpolated images and alpha values to the network
            d = self.network([x_alpha, alpha])

            # 6. Calculate the loss
            loss = tf.reduce_sum(tf.square(d - (x1 - x0)))

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Return loss values
        return {"loss": loss}

    def sample_iadb(self, x0, nb_step=128):
        """https://arxiv.org/abs/2305.03486, algorithm 4"""
        trajectory = [x0]
        x_alpha = x0
        bs = tf.shape(x_alpha)[0]
        for t in range(nb_step):
            alpha_start = t / nb_step
            alpha_end = (t + 1) / nb_step
            alpha_tensor = tf.fill((bs,), alpha_start)
            d = self.network([x_alpha, alpha_tensor], training=False)
            x_alpha = x_alpha + (alpha_end - alpha_start) * d
            trajectory.append(x_alpha)
        return x_alpha, trajectory
