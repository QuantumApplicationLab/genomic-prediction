# Quantum Inspired Genomic Prediction

This repository contains quantum inspired methods for evaluating and predicting genomic models. The algorithms found in this repository have been developed by the Quantum Application Lab

The project setup is documented in [project_setup.md](project_setup.md).

## Overview
Genomic evaluation and prediction are usually modelled using single nucleotide polymorphism best linear unbiased prediction (SNPBLUP) method. This model combines genomic and pedigree information about a sampled set of animals, and allows one to find the best animals to breed for a prioritised objective (i.e more milk yield for cows, higher egg yield for chickens, etc.). An example paper can be found [here](https://pure.tudelft.nl/ws/portalfiles/portal/89307446/s12711_021_00626_1.pdf). 

This repository uses the algorithms defined in the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) package, to evaluate this model using quantum inspired methods. 

## Set up

To install genomic_prediction from GitHub repository, do:

```console
$ git clone git@github.com:QuantumApplicationLab/genomic-prediction.git
$ cd genomic-prediction
$ python -m pip install .
```
or: 
```console
$ pip install https://github.com/QuantumApplicationLab/genomic-prediction.git
```

Then, one must clone the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) repository and installing the package locally or to a virtual environment:

```
$ git clone git@github.com:QuantumApplicationLab/quantum-inspired-algorithms.git
$ cd quantum-inspired-algorithms
$ python -m pip install .
```
or:
```console
$ pip install https://github.com/QuantumApplicationLab/quantum-inspired-algorithms.git
```

## Quantum Inspired Method

The quantum inspired method demonstrated has been based on,
- https://github.com/XanaduAI/quantum-inspired-algorithms
- "Quantum-inspired algorithms in practice", by Juan Miguel Arrazola, Alain Delgado, Bhaskar Roy Bardhan, and Seth Lloyd. 2020-08-13, volume 4, page 307. Quantum 4, 307 (2020).
- "Quantum-inspired low-rank stochastic regression with logarithmic dependence on the dimension", by András Gilyén, Seth Lloyd, Ewin Tang. (2018). ArXiv, abs/1811.04909.

The algorithm follows the basic following form:
1. Approximate SVD using FKV or HALKO (following https://arxiv.org/pdf/0909.4061) 
2. Estimate the coefficients using Monte Carlo methods
3. Sample the solution vector

## Examples

This repository contains experiments in the `test` folder. The experiments produced are defined in each file.   


## Contributing

If you want to contribute to the development of genomic_prediction,
have a look at the [contribution guidelines](CONTRIBUTING.md).

