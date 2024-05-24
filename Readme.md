# Diffusion GAN-based Oversampling Approach

Official implementation of "Diffusion GAN-based Oversampling for Imbalanced Tabular Data  ". Includes the code for diffusion GAN-based oversampling Approach for tabular data.

# Setup Instructions

## Install the required packages

 Python needs to be version 3.9+. Run the following to install a subset of necessary python packages for our code.

`
pip install -r requirements.txt
`

# Running the experiments

Here we describe the neccesary info for reproducing the experimental results.

## File structure

`models/` -- implementation of the proposed method.  

`train.py` is used to train DGOT using a given configs.

`evaluation.py` is used to evaluate DGOT on imbalanced data by using a evaluation model.

All main scripts are in `scripts/` folder:

- `scripts/evaluate_[binary|multi].py` are used to evaluate DGOT.
- `scripts/dataprocessing`.py -- related to data processing and loading.

Data folder (`datasets_prep/`):

The data used for this experiment are available through [UCI]([Datasets - UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)) , and we provide two sets of datasets as examples.

- `datasets_prep/[Data_name]` -- original data and related information as well as the same partition of the training/testing dataset as in the paper.

## Example

In this paper, we adopt the same set of hyper-parameters across all experiments for our method.   Template and example for the training of DGOT.

`python train.py --num_timesteps 4 --seed 666 --batch_size 512 --num_epoch 500 --exp exp0 --feature_len=52 --dataset thyroid_sick --class_num 2 --lr_d 8e-5 --lr_g 5e-3 --beta1 0.8 --beta2 0.9 --save_content_every 50 --save_ckpt_every 5 --save_content  --pw1 1 --pw2 1 --init_ch 16 --ch_mult 1 2 2 --nz 50 --rbg 4 --t_emb_dim 32 --ngf 32`

Template and example for the evaluation of DGOT.

`python evaluation.py --data_name thyroid_sick  --task_name binary --exps 0`
