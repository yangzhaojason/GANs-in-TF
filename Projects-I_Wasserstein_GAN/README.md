# GANs in Tensorflow (Multiple latest frameworks and projects)

## Projects I:  Wasserstein GAN

This is my implementation of the Wasserstein GAN algorithm (see the [paper](https://arxiv.org/abs/1701.07875)) in Tensorflow/Keras, as well as my takeaways from playing with it. 	We are drawing on the project of https://github.com/kuleshov/tf-wgan and maybe you are interested in the inverse convolution of the GAN network (**Deconvolution Networks**) model used in the article, see https://www.zhihu.com/question/43609045?sort=created

### Training a model

To train a model, use the `run.py` script:

```
python run.py train \
  --dataset <dataset> \
  --model <epochs> \
  --n-batch <batch_size> \
  --lr <learning_rate> \
  --c <c_parameter> \
  --n-critic <critic_steps> \
  --log-dir <log_dir> \
  --epochs <training_epochs>
```

The model will report training/validation losses in the logfile.

### Configuration

The repository contains code for a standard DC-GAN, trained using the usual GAN loss, as well as a Wasserstein GAN that uses a similar architecture.

The model parameters are configured via command-line options:

- `--model` is either `dcgan` or `wdcgan` (standard or Wasserstein)
- `--dataset` is `mnist` or our own data set (code example is 64✖64✖1)
- `--c` and `--n-critic` are the WGAN hyperparameters. The default values are the ones proposed in the paper.

Other flags are pretty self-explanatory.

