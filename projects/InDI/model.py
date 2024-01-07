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
"""
This is a reimplementation of InDI published in:
M. Delbracio and P. Milanfar:
"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration"
Transactions on Machine Learning Research (TMLR), 2023
https://arxiv.org/abs/2303.11435
"""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def setup_linear_mixture(a):
    """https://arxiv.org/abs/2303.11435, sec. 6.3"""
    # Create a mixture of two distributions: Uniform([0, 1]) and Delta(1)
    linear_mix = tfd.Mixture(
        cat=tfd.Categorical(probs=[1 / (1 + a), a / (1 + a)]),
        components=[
            tfd.Uniform(low=0., high=1.),
            tfd.Deterministic(loc=tf.constant(1., dtype=tf.float32)),
        ])

    return linear_mix


class InDIModel(tf.keras.Model):
    def __init__(self, network, rank=4, norm="L1", p_key="linear_0.5", eps=0.01, brownian_motion=False):
        super().__init__()
        self.network = network
        self.eps = tf.constant(eps)
        self.bm = brownian_motion
        self.p_t = self.setup_pt(p_key)
        self.loss_fn = self.get_loss_fn(norm)
        self.rank = rank  # support various data formats

    def broadcast_t(self, t):
        """broadcast to rank"""
        if self.rank == 2:  # e.g. 2d points
            return tf.expand_dims(t, axis=-1)
        elif self.rank == 4:  # e.g. RGB images
            return tf.expand_dims(tf.expand_dims(tf.expand_dims(t, axis=-1), axis=-1), axis=-1)
        else:
            raise NotImplementedError("Rank {} not yet supported".format(self.rank))

    def get_loss_fn(self, norm):
        """return corresponding loss function"""
        if norm == "L1":
            return tf.keras.losses.MeanAbsoluteError()
        elif norm == "L2":
            return tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError("Norm {} not yet implemented".format(self.norm))

    def setup_pt(self, p_key):
        """see https://arxiv.org/abs/2303.11435, sec. 3"""
        parts = p_key.split('_')
        k = parts[0]
        v = parts[1]
        if k == "linear":
            return setup_linear_mixture(float(v))
        else:
            raise NotImplementedError("key {} not yet implemented".format(self.p_key))

    def get_eps_t(self, t):
        """return eps_t"""
        if self.bm:
            # for numerical stability
            sqrt_t = tf.sqrt(tf.maximum(t, 1e-6))
            return self.eps / sqrt_t + 1e-6
        return self.eps

    def train_step(self, samples):
        # 1. Prepare y (noisy), x (high-quality)
        y, x = samples

        # 2. Sample t
        t = self.p_t.sample(tf.shape(y)[0])
        t_bc = self.broadcast_t(t)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the samples in the batch
            noise = tf.random.normal(shape=tf.shape(y), dtype=y.dtype)

            # 4. Obtain x_t (eq. 7)
            x_t = (1 - t_bc) * x + t_bc * y + t_bc * self.get_eps_t(t) * noise

            # 5. Pass the perturbed samples and timestamps to the network
            d = self.network([x_t, t])

            # 6. Calculate the loss -> eq. 9
            loss = self.loss_fn(x, d)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Return loss values
        return {"loss": loss}

    def sample_InDI(self, y, nb_step=500):
        """https://arxiv.org/abs/2303.11435, algorithm 1"""
        trajectory = [y]
        bs = tf.shape(y)[0]
        delta = 1 / nb_step

        # Sample gaussian noise
        noise = tf.random.normal(shape=tf.shape(y), dtype=y.dtype)

        # prepare \hat{x}_1
        x1 = y + self.get_eps_t(1.0) * noise
        xt_delta = x1
        counter = nb_step
        while counter > 0:
            t = counter / nb_step

            # Sample gaussian noise
            zeta = tf.random.normal(shape=tf.shape(x1), dtype=y.dtype)
            t_tensor = tf.fill((bs,), t)

            # note that
            # - s = t-delta
            # - if brownian_motion==True, noise perturbation is a pure Brownian motion,
            #   else noise corresponds to eq. 7 (zeta is canceled out)
            # - in practice, both methods appear to be equivalent (sec https://arxiv.org/abs/2303.11435, sec. 6.4),
            #   hence we stick to the simpler formulation by default.
            xt_delta = delta / t * self.network([xt_delta, t_tensor]) + (1 - delta / t) * xt_delta + (
                    t - delta) * tf.sqrt(self.get_eps_t(t - delta) ** 2 - self.get_eps_t(t) ** 2) * zeta
            trajectory.append(xt_delta)
            counter -= 1
        return xt_delta, trajectory
