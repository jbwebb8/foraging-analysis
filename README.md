# foraging-analysis
## Overview
This repository contains analysis code for the following publication:

Webb J, Steffan P, Hayden BY, Lee D, Kemere C, McGinley M (2025) Foraging animals use dynamic Bayesian updating to model meta-uncertainty in environment representations. PLoS Comput Biol 21(4): e1012989. https://doi.org/10.1371/journal.pcbi.1012989

The results of the statistical and behavioral models, as well as further analysis and derivations of their underlying mathematical behavior, can be found in iPython notebooks. Python files contain much of the heavy-lifting employed in the notebooks to provide a modular approach to the code organization and to make the notebooks themselves more user-friendly and interpretable. Additionally, they have several general-purpose toolkits for analysis and file-handling that may prove useful for other Python-based projects. 

The general structure of the repository is as follows:

- `config/`: Configuration file for `behavior.py`, a script for behavioral analysis summary. (archived)
- `docs/`: Useful resources for some pertinent mathematical constructs.
- `matlab/`: Scripts for data handling and analysis of head-fixed data. (archived)
- `notebooks/`: Primary directory for generating analysis results.
- `python/`: Primary directory for implementing file handling, analytical models, and other toolkits.

## Requirements
Python 3.7 or higher is required. A dedicated Python package manager, such as `virtualenv` or `anaconda`, is recommended. The Anaconda environment file, which contains package specifications and dependencies, can be found in `env.yaml`. Note this file contains a large number of dependencies due simultaneous development of other analytical and experimental tools; thus, some packages are not necessary for utilization of this repository alone. The iPython notebook (`.ipynb`) files can be loaded with the [Jupyter ecosystem](https://jupyter.org) (JupyterLab is recommended in order to access features such as table of contents). 

## Organizational framework
### Data analysis and results
The majority of the behavioral analyses for both the freely-moving and head-fixed tasks can be found in the `behavior_analysis.ipynb` notebook. The code is structured such that both datasets, though residing in different formats, are loaded into a common format, implemented by the `Session` class, in order to create a useful layer of abstraction. As mentioned, the [Table of Contents feature](https://jupyterlab.readthedocs.io/en/stable/user/toc.html) in JupyterLab is recommended for easier navigation (and in other large notebooks, such as `foraging_theory.ipynb`). The notebook contains:
- basic statistical analyses of behavioral metrics, such as residence time, travel time, and task-specific variables;
- statistical models of adaptations in residence time in response to environmental perturbations, including linear mixed models and cluster bootstrap models; and
- behavioral models of residence time in various environments, including heuristic, MVT, and Bayesian models.

Additionally, there are several notebooks not related to the published work. `ephys_preprocessing.ipynb` and `ephys_analysis.ipynb` format and analyze, respectively, electrophysiologic data from preliminary head-fixed foraging tasks. It may be useful for other projects that utilize the [MountainSort ecosystem](https://github.com/flatironinstitute/mountainsort). Note that these analyses depended on earlier versions of MountainSort that have now been archived; they may or may not be compatible with currently maintained versions. Regardless of the data ecosystem, `ephys_analysis.ipynb` contains a variety of general analyses, including peristimulis time histograms (PSTHs) and dimensionality reduction techniques (e.g. PCA, PPCA, FA, and GPFA), that are extendable to many neural datasets. In keeping with the structure of this repository, much of the heavy-lifting for the algorithms is offloaded to Python files, particularly `ephys.py`. The notebooks `patch_foraging_lt.ipynb` and `patch_foraging_simulation.ipynb` were related to earlier versions of the foraging task and are no longer pertinent.

### Theoretical framework
The details of the reward process underlying the foraging task are discussed in `poisson_drip.ipynb`, which includes validation of the methods used to generate and analyze reward timing. The behavioral models of foraging in environments with this underlying reward process are characterized in two broad categories. The global approach, wherein animals estimate the average statistics of the environment to make foraging decisions, is discussed in `mvt.ipynb`; it outlines the conceptual framework of the marginal value theorem [1], as well as analyses that guided the choice of task variables in the foraging tasks. Models with a local component (that is, a Bayesian estimator using recent observations) are developed in `foraging_theory.ipynb`, which contains full derivations of the equations in the publication. Additionally, the notebook begins with an analysis of a related work [2] on which the model is based and derives the relationship with the published model.

Lastly, `models.ipynb` develops and tests many of the distribution models used throughout this repository. Additionally, it tests the implementation of the cluster bootstrap method used in the publication.

### Miscellaneous utilities
Videos of the freely-moving behavior were recorded using Logitech C920 webcams. Initial analyses of preliminary data, which guided the design of the experimental setup, are found in `video_analysis.ipynb`. In addition to camera positioning and synchronization, the notebook contains a workflow for position tracking using DeepLabCut, a deep neural network that can label body parts, and by proxy position, across video frames.

The experimental rig for the head-fixed task necessitated additional steps to sychronize treadmill and behavioral data. The algorithm, and its validation, are shown in `wheel_alignment.ipynb`.

The notebook `playground.ipynb` is, as suggested by its name, a collection of code snippets that were useful for data processing and debugging throughout the behavioral analyses.

## References
1. Stephens, David W., and John R. Krebs. Foraging Theory. Princeton University Press, 1986. https://press.princeton.edu/titles/2453.html.
2. Kilpatrick, Zachary P., Jacob D. Davidson, and Ahmed El Hady. “Normative Theory of Patch Foraging Decisions.” ArXiv:2004.10671 [Math, q-Bio], April 22, 2020. http://arxiv.org/abs/2004.10671.
