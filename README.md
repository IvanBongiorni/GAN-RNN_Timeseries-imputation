Author: **Ivan Bongiorni**, Data Scientist. [LinkedIn](https://www.linkedin.com/in/ivan-bongiorni-b8a583164/).

# Convolutional Recurrent Seq2seq GAN for the Imputation of Missing Values in Time Series Data

<a href="url" align="center"><img src="https://github.com/IvanBongiorni/GAN-RNN_Timeseries-imputation/blob/master/utils/imputation_example_00.png" align="center" height="204" width="800" ></a>
<a href="url" align="center"><img src="https://github.com/IvanBongiorni/GAN-RNN_Timeseries-imputation/blob/master/utils/imputation_example_02.png" align="center" height="204" width="800" ></a>

## Description

The goal of this project is the implementation of multiple configurations of a **Recurrent Convolutional Seq2seq** neural network for the imputation of time series data. Three implementations are provided:

1. A **Recurrent Convolutional seq2seq** model.
2. A **GAN** (*Generative Adversarial Network*) based on the same architecture above, where an Imputer is trained to fool an adversarial Network that tries to distinguish real and fake (imputed) time series.
3. A **partially adversarial model**, in which both Loss structures of previous models are combined in one: an Imputer model must reduce true error Loss, while trying to fool a Discriminator at the same time.

Models are Implemented in TensorFlow 2 and trained on the [Wikipedia Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset.

<a href="url" align="center"><img src="https://github.com/IvanBongiorni/GAN-RNN_Timeseries-imputation/blob/master/utils/performance_comparison_3models.png" align="center" height="300" width="800" ></a>

<br/>

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

Notebooks and explanations:
- `how_it_works.md`: contains explanation of Deep Learning models in greater detail.
- `nan_exploration.ipynb`: contains a study of the distribution of NaN's in the raw dataset, that lead to the development of the deterioration function.
- `data_scaling_exploration.ipynb`: contains visualizations of the scaling function I employed in data preprocessing phase.
- `imputation_visual_check.ipynb`: visualization of a models performance. The notebook loads the trained model specified in `params['model_name']` and check its performance on Validation and Test data.
- `performance_comparison.ipynb`: shows the performances of three trained models on Test data, compared.

Folders:
- `data_raw/`: it is supposed to contain the raw Wikipedia Web Traffic Time Series Forecasting dataset, as it is downloaded (and unzipped) from Kaggle.
- `data_processed/`: it contains the outcome of preprocesing pipeline, launched from `main_processing.py`. Observations will be stored in three sub-directories for `Training/`, `Validation/` and `Test/`.
- `saved_models/`: where models are saved at the end of training pipepine. Model names can be changed in `config.yaml`. In case a GAN is trained and config parameter `save_discriminator` is set to `True`, the Discriminator model will be saved as `[model_name]_discriminator.h5`.

<br/>

## Modules required

```
langdetect==1.0.8
numpy==1.18.3
pandas==1.0.3
scikit-learn==0.22.2.post1
scipy==1.4.1
tensorflow==2.1.0
```
<br/>

## Bibliography
- *Luo, Y., Cai, X., Zhang, Y., & Xu, J. (2018). Multivariate time series imputation with generative adversarial networks. In Advances in Neural Information Processing Systems (pp. 1596-1607).*
- *Yoon, J., Jordon, J., & Van Der Schaar, M. (2018). Gain: Missing data imputation using generative adversarial nets. arXiv preprint arXiv:1806.02920.*
- *Guo, Z., Wan, Y., & Ye, H. (2019). A data imputation method for multivariate time series based on generative adversarial network. Neurocomputing, 360, 185-197.*
- *Liu, Y., Yu, R., Zheng, S., Zhan, E., & Yue, Y. (2019). NAOMI: Non-autoregressive multiresolution sequence imputation. In Advances in Neural Information Processing Systems (pp. 11238-11248).*
- *Luo, Y., Zhang, Y., Cai, X., & Yuan, X. (2019, August). E2GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation. In Proceedings of the 28th International Joint Conference on Artificial Intelligence (pp. 3094-3100). AAAI Press.*
- *Suo, Q., Yao, L., Xun, G., Sun, J., & Zhang, A. (2019, June). Recurrent Imputation for Multivariate Time Series with Missing Values. In 2019 IEEE International Conference on Healthcare Informatics (ICHI) (pp. 1-3). IEEE.*
- *Tang, X., Yao, H., Sun, Y., Aggarwal, C. C., Mitra, P., & Wang, S. (2020). Joint Modeling of Local and Global Temporal Dynamics for Multivariate Time Series Forecasting with Missing Values. In AAAI (pp. 5956-5963).*
- *Zhang, J., Mu, X., Fang, J., & Yang, Y. (2019). Time Series Imputation via Integration of Revealed Information Based on the Residual Shortcut Connection. IEEE Access, 7, 102397-102405.*
- *Fortuin, V., Baranchuk, D., Rätsch, G., & Mandt, S. (2020, June). GP-VAE: Deep Probabilistic Time Series Imputation. In International Conference on Artificial Intelligence and Statistics (pp. 1651-1661).*
- *Huang, T., Chakraborty, P., & Sharma, A. (2020). Deep convolutional generative adversarial networks for traffic data imputation encoding time series as images. arXiv preprint arXiv:2005.04188.*
- *Huang, Y., Tang, Y., VanZwieten, J., & Liu, J. (2020). Reliable machine prognostic health management in the presence of missing data. Concurrency and Computation: Practice and Experience, e5762.*
- *Jun, E., Mulyadi, A. W., Choi, J., & Suk, H. I. (2020). Uncertainty-Gated Stochastic Sequential Model for EHR Mortality Prediction. arXiv preprint arXiv:2003.00655.*
- *Qi, M., Qin, J., Wu, Y., & Yang, Y. (2020). Imitative Non-Autoregressive Modeling for Trajectory Forecasting and Imputation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12736-12745).*
- *Wang, Y., Menkovski, V., Wang, H., Du, X., & Pechenizkiy, M. (2020). Causal Discovery from Incomplete Data: A Deep Learning Approach. arXiv preprint arXiv:2001.05343.*
- *Yi, J., Lee, J., Kim, K. J., Hwang, S. J., & Yang, E. (2019). Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks. arXiv preprint arXiv:1906.00150.*
- *Yoon, S., & Sull, S. (2020). GAMIN: Generative Adversarial Multiple Imputation Network for Highly Missing Data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8456-8464).*

<br/>

## Hardware
I trained this model on a fairly powerful machine: a System76 Adder WS laptop with 64 GB of RAM and NVidia RTX 2070 GPU.
