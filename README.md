# IMR
The experimental database and code are stored at https://github.com/YongYu123/IMR.

The experiment runs in an Anaconda environment with the following dependencies:

pytorch 1.9.0

numpy 1.21.5


The Jupyter folder contains the .ipynb format code, with subfolders corresponding to the Moving MNIST and Sediment distribution datasets.

For Moving MNIST, you need to first execute the commands in generate-moving_minst.ipynb to generate a Moving MNIST dataset. Then execute bi_mdn-rnn-imputation.ipynb, which contains code for generating random mask files for other imputation methods.

For Sediment distribution, you need to first download the sediment distribution database from the Google drive link https://drive.google.com/file/d/1BjPFe4SB18u_O64DbSAD_wTaaEtEBOZn/view?usp=sharing to the Sediment Distribution folder, then execute bi_mdn-rnn-imputation.ipynb to generate a random mask for other imputation methods.
