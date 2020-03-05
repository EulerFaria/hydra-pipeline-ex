# A framework for Data Science Experiments
![MKT](https://img.shields.io/badge/version-v0.1-blue.svg)
![MKT](https://img.shields.io/badge/language-Python-orange.svg)

# Description

This repository contains a basic application of [Hydra](https://hydra.cc/docs/) from Pytorch and [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) from Sklearn to facilitate reproducibility in Data Science projects during the experimentation cycle. It uses Hydra to manage a series of `config` files, to generate automatically the directories containing the outputs of each experiment, and to compose a new set of experiments by using the flag `--multirun`. Also, it takes advantage of Pipeline to assemble several steps that can be cross-validated together while setting different parameters. In this project it is used just two types of scaling methods: [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) and a [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the dimensionality of the feature matrix.


All of those steps were **randomly chosen to serve as examples** of possible configurations to emphasize the capabilities of Hydra and Pipeline together. The dataset used for running the experiments is the Breast Cancer from sklearn which is a binary classification problem.


# Features 

- Reproducibility of Data Science experiments;
- Config management;
- Basic Logging;
- Performance Report for binary classification;
- Hyperparameter optimization through GridSearchCV; 

# Environment configuration

The packages used to develop the project are listed in the `environment.yml` file. To reproduce the results and functionalities of the project you should install  [Anaconda](https://anaconda.org) and set up the environment as explained in what follows. However, if you prefer you can also create your enviroment using `pip env` and install the required libraries.



## Conda env

Once you have installed Anaconda, open a terminal and go to the directory of the project. 

```
$ cd .../HYDRA-PIPELINE-EX 

```

After that run the following command

```
$ conda env create -f environment.yml

```
Then activate your new environment using the following command :

```
$ conda activate hydra_pipe

```
Finally check the version of the libraries

```
$ conda list

```
# Running the example 

Once you have installed and activated the conda env, just run the following command in the terminal for one experiment. 

```
$ python main.py 

```
This command will apply the `defaults` parameters of `conf/config.yaml`.

The other option is to run the following command:

```
$ python main.py --multirun scaling=std,minmax

```
The above comand will create new experiments using the different scaling methods provided in this example. 

Stick to the concepts,you can take the main idea behind this example and adjust it according with your needs. 

# Next Steps

- Implement errror handling with try and except clauses and feed the logger;
- Implement different Hyperparameter optimization methodos such as RandomSearchCV, BayesianOptimization;
- Add more models in to `conf/model/*.yaml` files  


Every feedback is welcomed. 

 # Authors: 
 - [EulerFaria](https://github.com/EulerFaria);
 - [Anderson Vinco]();
 