# Seq2seq GAN for the Imputation of Missing Values in Time Series Data

Author: Ivan Bongiorni


## Description

The goal of this project is the implementation of multiple configurations of a **Recurrent Convolutional Seq2seq** neural network for the imputation of time series data. Three implementations are provided:

1. A "Vanilla" seq2seq model.
2. A GAN (Generative Adversarial Network), where an Imputer is trained to fool an adversarial Network that tries to distinguish real and fake (imputed) time series.
3. A partially adversarial model, in which both Loss structures of previous models are combined in one: an Imputer model must reduce true error Loss, while at the same time try to fool a Discriminator.

Their performance the models and their ensembles is then compared, together with simpler imputation methods for comparison.

Models are Implemented in TensorFlow 2 and trained on the [Wikipedia Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset.


## Files
- `config.yaml`: configuration parameters for data preprocessing, training and testing.

Pipelines:
- `main_processing.py`: starts data preprocessing pipeline. Its outcomes are ready-to-train datasets saved in .npy (`numpy`) format in `/data_processed/` folder.
- `main_train.py`: starts training pipeline. Trained model is saved in `/saved_models/` folder, with the '`model_name`' provided in `config.yaml`.

Scripts:
- `tools.py`: contains more technical functions that are iterated during preprocessing pipeline.
- `model.py`: implementation of models' architectures.
- `train.py`: contains functions for all training configurations.
- `deterioration.py`: the script contains the function that calls an artificial deterioration of training data, in order to check imputation performance.
- `impute.py`: final script, to be called in order to produce imputed data (for raw time series that contain NaN's) and export them for future projects.

Notebooks and explanations:
- `how_it_works.md`: contains explanation of Deep Learning models in greater detail.
- GAN_vs_others.ipynb  # to be renamed: `performance_comparison.ipynb`
- `data_scaling_exploration.ipynb`: contains visualizations of the scaling function I employed in data preprocessing phase.
- `nan_exploration.ipynb`: contains a study of the distribution of NaN's in the raw dataset, that lead to the development of the deterioration function.

Folders:
- `/data_raw`: this directory is supposed to contain the raw Wikipedia Web Traffic Time Series Forecasting dataset, as it is downloaded (and unzipped) from Kaggle.
- `/data_processed`: contains the outcome of preprocesing pipeline, launched from `main_processing.py`. Contains three sub-directories for `/Train/`, `/Validation/` and `/Test/` observations.
- `/saved_model`: where all saved models are saved at the end of training pipepine. Model names can be changed in `config.yaml`. In case a GAN is trained and config parameter `save_discriminator` is set to `True`, the Discriminator model will be saved as `[model_name]_discriminator`.


## Modules required

```
langdetect==1.0.8
numpy==1.18.3
pandas==1.0.3
scikit-learn==0.22.2.post1
scipy==1.4.1
tensorflow==2.1.0
```

## Hardware

I trained this model on a fairly powerful machine: a System76 Adder WS laptop with 64 GB of RAM and an NVidia RTX 2070 GPU.
