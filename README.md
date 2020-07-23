# A simple framework for Data Science Experiments

![MKT](https://img.shields.io/badge/version-v0.2-blue.svg)
![MKT](https://img.shields.io/badge/language-Python-orange.svg)

# Description

This repository contains a basic application of [Hydra](https://hydra.cc/docs/intro) from Pytorch, [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) from Sklearn to facilitate reproducibility in Data Science projects during the experimentation cycle. It uses Hydra to manage a series of `config` files, to generate automatically the directories containing the outputs of each experiment, and to compose a new set of experiments by using the flag `--multirun`. Also, it takes advantage of Pipeline to assemble several steps that can be cross-validated together while setting different parameters. 

For demonstration purposes, it was used two types of scaling methods: [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler), and two estimators [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). 

Notice that with the functions provided in the module `hydra_pipeline`you can easily add any step you want in your Pipeline by just creating another config module following the pattern used. It is also provided a Class wrapping the Bayesian Optimization as [sklearn.model_selection._search.BaseSearchCV](https://github.com/scikit-learn/scikit-learn/blob/effc4364508288b0410967c5c2786f1b27b76185/sklearn/model_selection/_search.py#L402) to make it the same pattern of other hyperparameters optimization methods provided by Sklearn.

# Features

- Reproducibility of Data Science experiments;
- High flexibility for adding any pipeline step based on the name of the package;
- Basic Logging;
- Performance Report for binary classification;
- Hyperparameter optimization through Bayesian Optimization,GridSearchCV and RandomSearchCV;

# Environment configuration

The packages used to develop the project are listed in the `environment.yml` file. To reproduce the results and functionalities of the project you should install [Anaconda](https://anaconda.org) and set up the environment as explained in what follows. However, if you prefer you can also create your enviroment using `pip env` and install the required libraries.

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
$ python train.py

```

This command will apply the `defaults` parameters of `conf/config.yaml`.

The other option is to run the following command:

```
$ python train.py --multirun scaling=std,minmax model=rf,svc

```

The above comand will create new experiments using the different scaling methods provided in this example.

Stick to the concepts,you can take the main idea behind this example and adjust it according with your needs.


# Authors:

Every feedback is welcomed.

- [Euler Faria](https://github.com/EulerFaria);
- [Alan de Aguiar](https://github.com/alanAguiar);
- [Anderson Vinco](https://github.com/AndersonFVinco);
