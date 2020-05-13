# GAN-RNN_Timeseries-imputation

## WORK IN PROGRESS

The goal of this project is the implementation of multiple configurations of a Recurrent Convolutional Seq2seq model for the imputation of time series data. Three implementations are provided:

0. A "Vanilla" seq2seq model.
0. A GAN (Generative Adversarial Network), where an Imputer is trained to fool an adversarial Network that tries to distinguish real and fake (imputed) time series.
0. A partially adversarial model, in which the both Loss structure of previous models are comined in one: an Imputer model must reduce true error Loss, while at the same time try to fool a Discriminator.

Their performance the models and their ensembles is then compared, together with simpler imputation methods for comparison.

Models are Implemented in TensorFlow 2 and trained on the [Wikipedia Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset.


## Files
- `how_it_works.md`: contains explanation of Deep Learning models in greater detail.
- `config.yaml`: configuration parameters for data preprocessing, training and testing.

- `main_processing.py`: starts data preprocessing pipeline. Its outcomes are ready-to-train datasets saved in .npy (`numpy`) format in `/data_processed/` folder.
- `tools.py`: contains more technical functions that are iterated during preprocessing pipeline.
- `main_train.py`: starts training pipeline. Trained model is saved in `/saved_models/` folder, with the '`model_name`' provided in `config.yaml`.
- `model.py`: implementation of models' architectures.
- `train.py`: contains functions for all training configurations.
- `deterioration.py`: the script contains the function that calls an artificial deterioration of training data, in order to check imputation performance.
- `impute.py`: final script, to be called in order to produce imputed data (for raw time series that contain NaN's) and export them for future projects.

### Notebooks
- GAN_vs_others.ipynb
- data_scaling_exploration.ipynb
- nan_exploration.ipynb


## Modules reuired

```
langdetect==1.0.8
numpy==1.18.3
pandas==1.0.3
scikit-learn==0.22.2.post1
scipy==1.4.1
tensorflow==2.1.0
```
