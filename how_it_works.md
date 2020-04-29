Author: Ivan Bongiorni
2020-04-28
Repository:

# Convolutional Recurrent Seq2seq with Adversarial Training for Missing Data Imputation of Time Series
# How it works

## Organization of the Dataset


## Train-Validation-Test split

After the removal of observations that originally came with NaN's, I have split my data "vertically": I reserved some whole trends to Train, Validation, and Test sets respectively. 
No series present in one of these three subsets has any trace in the others. The reason for this choice is that the goal of this project is to be able to impute *real* missing observations as good as possible.
Therefore, I needed to test my model's ability to imput data from trends it had never seen before.


## Training an ensemble
