<h1 align="center">Louvain and Leiden Algorithms</h1>

<p align="center">
    A python implementation of the Louvain and Leiden community detection algorithms.
    <br> 
</p>

---

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
- [Sample Notebooks](#sample-notebooks)
- [Tests and Static Analysis](#tests-and-static-analysis)
- [Sources](#sources)

## About
This project is an implementation of the [Louvain][src-blondel] and [Leiden][src-traag] algorithms for [community detection](https://en.wikipedia.org/wiki/Community_structure) in graphs.  
The implementation was conducted and tested using [Python](https://python.org) version 3.10.12.

## Getting Started
In order to try out this implementation, do the following:
1. Check out this repository, or [download the latest version](https://git.esclear.de/esclear/louvain-leiden/archive/main.zip) and unzip it somewhere.
2. Install the [dependencies](#dependencies), for example using `pip install -r requirements.txt`.
3. Run `jupyter notebook` and explore the available [notebooks](#sample-notebooks).

### Dependencies
To use this implementation, you only need the [NetworkX](https://networkx.org/) graph library – it was implemented and tested with version 3.0:
```
networkx==3.0
```
For running the notebooks and working with the example datasets, you will need the following packages in addition to NetworkX:
```
jupyter_core==5.2.0
notebook==6.5.2
pandas==1.5.3
matplotlib==3.7.0
```
Lastly, to run the tests, generate code coverage information and run static analysis, the following packages are required:
```
pytest==7.2.1
pytest-cov==4.0.0
black==23.1.0
mypy==1.0.1
ruff==0.0.269
```

The python dependencies are specified in the [`requirements.txt`](requirements.txt) file and can be installed (preferrably in a virtual environment) using the command `pip install -r requirements.txt`.  
Alternatively, if you're running [nix](https://nixos.org) and if you are so inclined, you can also run `nix-shell` to install the dependencies.

## Sample Notebooks
For demonstration purposes of this library, the following notebooks are included in this repository:
- [`Demo (Karate Club).ipynb`](Demo%20(Karate%20Club).ipynb), which demonstrates the usage of this implementation,
- [`Performance Comparison.ipynb`](Performance%20Comparison.ipynb), which compares our implementation of the Louvain algorithm to the implementation in the NetworkX library, and
- [`Parameter Exploration.ipynb`](Parameter%20Exploration.ipynb), which examines the influence of parameters on the detected communities on a larger problem instance.

## Tests and Static Analysis

### Running the tests
This project has unit tests for almost all individual [utility functions](community_detection/utils.py) and the [quality functions](community_detection/quality_functions.py).
In addition to that, there are tests which execute the community detection algorithms on known graphs and check the results.

The tests are located in the [`tests`](tests/) directory and can be executed running the `pytest` tool as follows from a shell in the repository root, generating [branch coverage](https://en.wikipedia.org/wiki/Code_coverage#Basic_coverage_criteria) information in the process:

```bash
pytest --cov --cov-report=html
```
The coverage information will be placed in a directory called `htmlcov`.

### Running static code analysis
Apart from the tests, this project employs a number of static analysis tools:

#### Linter: `ruff`
The tool **ruff** is used to lint the code and look for any problems with the code.
A variety of checks is [configured](pyproject.toml), among them lints which check for public functions having documentation blocks   in the correct format, all functions being annotated with type information, etc.  
To check that the code satisfies the linting rules, run the following command:
```bash
ruff .
```
It will list problems it has found in the code.
If no lint finds an issue, then this command produces no output.

#### Formatter: `black`
The **black** formatter ensures that the python code is formatted according to [its code style](https://black.readthedocs.io/en/stable/the_black_code_style/index.html).
To reformat the code according to the code style, run the following command:
```bash
black .
```
You should see the line `14 files left unchanged.` when running it on a fresh checkout.

#### Type checker: `mypy`
Python is a dynamically typed language, meaning that the type of a variable may change over its lifetime and that type checking is done at runtime.  
The code uses [type hints](https://peps.python.org/pep-0483/) for all functions, in order to add static types, which can be checked using a type checker, such as `mypy`.
The type hints serve as further documentation of a function's usage and allow for a type checker to verify the types of variables and functions in a program, so that it may calls to functions with arguments of an incorrect type, for example.  
To check this project using `mypy`, run the following command:
```bash
mypy .
```
You should see the message `Success: no issues found in 14 source files`.

## Sources
The following papers were used as sources for the implementation:
- [_Fast unfolding of communities in large networks_][src-blondel] for a description of the Louvain algorithm.
- [_From Louvain to Leiden: guaranteeing well-connected communities_][src-traag] and especially its [supplementary material][src-traag-supp] for pseudocode for the algorithms and some further explanations of the ideas.

The notebooks in this repository use the following datasets:
- The [Karate Club][data-karateclub] dataset (provided by the NetworkX library)
- The [Cora][data-cora] dataset (contained in the [datasets](datasets) directory)
- The [Jazz Musicians][data-jazz] dataset (also part of the [datasets](datasets) directory)

[src-blondel]: https://doi.org/10.1088/1742-5468/2008/10/p10008 "Blondel, Guillaume, Lambiotte, Lefebvre: Fast unfolding of communities in large networks"
[src-traag]: https://doi.org/10.1038/s41598-019-41695-z "Traag, Waltman, van Eck: From Louvain to Leiden: guaranteeing well-connected communities"
[src-traag-supp]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-019-41695-z/MediaObjects/41598_2019_41695_MOESM1_ESM.pdf "Traag, Waltman, van Eck: Supplementary Material to: From Louvain to Leiden: guaranteeing well-connected communities"
[data-jazz]: https://www.worldscientific.com/doi/abs/10.1142/S0219525903001067 "Gleiser, Danon: Community structure in jazz"
[data-karateclub]: https://www.journals.uchicago.edu/doi/pdf/10.1086/jar.33.4.3629752 "An Information Flow Model for Conflict and Fission in Small Groups"
[data-cora]: https://www.openicpsr.org/openicpsr/project/100859/version/V1/view "McCallum: Cora Dataset"
[ruff]: https://ruff.rs/
