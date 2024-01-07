# Diffusion-TF

This repository provides a collection of TensorFlow 2 implementations/ ports for diffusion-based research. 

The projects are self-contained and can be found [here](https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/). 
All models are instances of [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), hence multi-gpu training can be enabled with minor adjustments thanks to [tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_custom_training_loops).

Under active development.

## Updates

***01/07/2024***

1. Initial release of this project.
2. Released  [Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model](https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/IADB) by Heitz et al., in Special Interest Group on Graphics and Interactive Techniques (Siggraph), 2023.
3. Released [Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration](https://github.com/Nikolai10/Diffusion-TF/blob/master/projects/InDI) by Delbracio & Milanfar, in Transactions on Machine Learning Research (TMLR), 2023.

## License
[Apache License 2.0](LICENSE)
