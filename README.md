# Quantum inspired genomic_prediction

This repository contains quantum inspired methods for evaluating and predicting genomic models. 

The project setup is documented in [project_setup.md](project_setup.md).

## Genomic evaluation
Genomic evaluation and prediction are usually modelled using single nucleotide polymorphism best linear unbiased prediction (SNPBLUP) method. This model combines genomic and pedigree information about a sampled set of animals, and allows one to find the best animals to breed for a prioritised objective (i.e more milk yield for cows, higher egg yield for chickens, etc.). An example paper can be found [here](https://pure.tudelft.nl/ws/portalfiles/portal/89307446/s12711_021_00626_1.pdf). 

This repository uses the algorithms defined in the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) package, to evaluate this model using quantum inspired methods. 

## Set up

To install genomic_prediction from GitHub repository, do:

```console
git clone git@github.com:QuantumApplicationLab/genomic-prediction.git
cd genomic-prediction
python -m pip install .
```

Then, one must clone the [quantum-inspired-algorithms](https://github.com/QuantumApplicationLab/quantum-inspired-algorithms) repository and installing the package locally or to a virtual environment:

```
git clone git@github.com:QuantumApplicationLab/quantum-inspired-algorithms.git
cd quantum-inspired-algorithms
python -m pip install .
```

## Examples

This repository contains experiments in the `test` folder. The tests are defined in each file. 


## Contributing

If you want to contribute to the development of genomic_prediction,
have a look at the [contribution guidelines](CONTRIBUTING.md).

