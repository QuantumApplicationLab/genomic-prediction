# Quantum-Inspired Genomic Prediction

This repository contains quantum-inspired methods for constructing models for genomic prediction. These have been developed by the Quantum Application Lab.

The project setup is documented in [project_setup.md](project_setup.md).

## Overview
Genomic prediction is usually performed using the single nucleotide polymorphism best linear unbiased prediction (SNPBLUP) method. This method combines genomic and pedigree information about a sampled set of animals, and allows one to find the best animals to breed for a given objective (e.g., milk or egg yield). An example paper can be found [here](https://pure.tudelft.nl/ws/portalfiles/portal/89307446/s12711_021_00626_1.pdf). 

This repository leverages the algorithms in the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) package.

## Setup

To install genomic_prediction from GitHub repository, do:

```shell
$ git clone git@github.com:QuantumApplicationLab/genomic-prediction.git
$ cd genomic-prediction
$ python -m pip install .
```
or: 
```shell
$ pip install https://github.com/QuantumApplicationLab/genomic-prediction.git
```

Then, clone the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) repository and install the package locally or on a virtual environment:

```shell
$ git clone git@github.com:QuantumApplicationLab/quantum-inspired-algorithms.git
$ cd quantum-inspired-algorithms
$ python -m pip install .
```
or:
```shell
$ pip install https://github.com/QuantumApplicationLab/quantum-inspired-algorithms.git
```

## Quantum-Inspired Method

The quantum-inspired method has been based on:
- https://github.com/XanaduAI/quantum-inspired-algorithms
- "Quantum-inspired algorithms in practice", by Juan Miguel Arrazola, Alain Delgado, Bhaskar Roy Bardhan, and Seth Lloyd. 2020-08-13, volume 4, page 307. Quantum 4, 307 (2020).
- "Quantum-inspired low-rank stochastic regression with logarithmic dependence on the dimension", by András Gilyén, Seth Lloyd, Ewin Tang. (2018). ArXiv, abs/1811.04909.

The algorithm follows the following form:
1. Approximate SVD using FKV or [HALKO](https://arxiv.org/pdf/0909.4061).
2. Estimate coefficients using Monte Carlo methods.
3. Sample the solution vector.

## Examples

Quantum-inspired experiments can be found in `tests/test_quantum_inspired_small.py`. For example, in `test_qi_no_X`, breeding values are estimated without fixed effects.
In particular, this experiment generates visualizations of the animals predicted to be among the top fifty using Halko's method. Below is an example of when 1000 Monte Carlo iterations are used. Only the top 4 animals are correctly selected out of the top 50.
![1000](docs/test_qi_no_X_1000_matches.png)
However, when 100,000 Monte Carlo iterations are used, 15 animals are correctly selected out of the top 50. 
![100000](docs/test_qi_no_X_100000_matches.png)

## Contributing

If you want to contribute to the development of genomic_prediction,
have a look at the [contribution guidelines](CONTRIBUTING.md).
