# Upcoming release

# Release 0.11.0
* Add expectation-maximisation (EM) algorithm to learn with latent variables
* Fix infinite loop when querying `InferenceEngine` after a do-intervention that splits
  the graph into two or more subgraphs, as per #45 and #100
* Allow users to provide self-defined CPD, as per #18 and #99
* Fix decision tree and mdlp discretisations bug when input data is shuffled
* Fix broken URLs in FAQ documentation, as per #113 and #125
* Add a link to `PyGraphviz` installation guide under the installation prerequisites
* Fix integer index type checking for timeseries data, as per #74 and #86
* Add GPU support to Pytorch implementation, as requested in #56 and #114
* Add an example for structure model exporting into first causalnex tutorial, as per #124 and #129

# Release 0.10.0
* Add supervised discretisation strategies using Decision Tree and MDLP algorithms
* Add `BayesianNetworkClassifier` an sklearn compatible class for fitting and predicting probabilities in a BN
* Fixes cyclical import of `causalnex.plots`, as per #106
* Add utility function to extract Markov blanket from a Bayesian Network
* Support receiving a list of inputs for `InferenceEngine` with a multiprocessing option
* Add supervised discretisation strategies using Decision Tree and MDLP algorithms
* Added manifest files to ensure requirements and licenses are packaged
* Fix estimator issues with sklearn ("unofficial python 3.9 support", doesn't work with `discretiser` option)
* Minor bumps in dependency versions, remove prettytable as dependency

# Release 0.9.2
* Remove Boston housing dataset from "sklearn tutorial", see #91 for more information.
* Update pylint version to 2.7
* Improve speed and non-stochasticity of tests

# Release 0.9.1
* Fixed bug where the sklearn tutorial documentation wasn't rendering.
* Weaken pandas requirements to >=1.0, <2.0 (was ~=1.1).

# Release 0.9.0
* Removed Python 3.5 support and add Python 3.8 support.
* Updated core dependencies, supporting pandas 1.1, networkx 2.5, pgmpy 0.1.12.
* Added PyTorch to requirements (i.e. not optional anymore).
  * Allows sklearn imports via `from causalnex.structure import DAGRegressor, DAGClassifier`.
* Added multiclass support to pytorch sklearn wrapper.
* Added multi-parameter collapsed graph as graph attribute.
* Added poisson regression support to sklearn wrapper.
* Added distribution support for structure learning:
  * Added ordinal distributed data support for pytorch NOTEARS.
  * Added categorical distributed data support for pytorch NOTEARS.
  * Added poisson distributed data support for pytorch NOTEARS.
* Added dist type schema tutorial to docs.
* Updated sklearn tutorial in docs to show new features.
* Added constructive ImportError for pygraphviz.
* Added matplotlib and ipython display convenience functions.

# Release 0.8.1

* Added `DAGClassifier` sklearn interface using the Pytorch NOTEARS implementation. Supports binary classification.
* Added binary distributed data support for pytorch NOTEARS.
* Added a "distribution type" schema system for pytorch NOTEARS (`pytorch.dist_type`).
* Rename "data type" to "distribution type" in internal language.
* Fixed uniform discretiser (`Discretiser(method='uniform')`) where all bins have identical widths.
* Fixed and updated sklearn tutorial in docs.

# Release 0.8.0

* Added DYNOTEARS (`from_numpy_dynamic`, an algorithm for structure learning on Dynamic Bayesian Networks).
* Added Pytorch implementation for NOTEARS MLP (`pytorch.from_numpy`) which is much faster and allows nonlinear modelling.
* Added `DAGRegressor` sklearn interface using the Pytorch NOTEARS implementation.
* Added non-linear data generators for multiple data types.
* Added a count data type to the data generator using a zero-inflated Poisson.
* Set bounds/max class imbalance for binary features for the data generators.
* Bugfix to resolve issue when applying NOTEARS on data containing NaN.
* Bugfix for data_gen system. Fixes issues with root node initialization.

# Release 0.7.0

* Added plotting tutorial to the documentation
* Updated `viz.draw` syntax in tutorial notebooks
* Bugfix on notears lasso (`from_numpy_lasso` and `from_pandas_lasso`) where the non-negativity constraint was not being set
* Added DAG-based synthetic data generator for mixed types (binary, categorical, continuous) using a linear SEM approach.
* Unpinned some requirements

# Release 0.6.0

* support for newer versions of scikit-learn
* classification report now returns dict in line with scikit-learn

# Release 0.5.0

* Plotting now backed by pygraphviz. This allows:
   * More powerful layout manager
   * Cleaner fully customisable theme
   * Out-the-box styling for different node and edge types
* Can now get subgraphs from StructureModel containing a specific node
* Bugfix to resolve issue when fitting CPDs with some missing states in data
* Minor documentation fixes and improvements

# Release 0.4.3:

Bugfix to resolve broken links in README and minor text issues.

# Release 0.4.2:

Bugfix to add image to readthedocs

# Release 0.4.1:

Bugfix to address readthedocs issue.

# Release 0.4.0:

The initial release of CausalNex.

## Thanks for supporting contributions
CausalNex was originally designed by [Paul Beaumont](https://www.linkedin.com/in/pbeaumont/) and [Ben Horsburgh](https://www.linkedin.com/in/benhorsburgh/) to solve challenges they faced in inferencing causality in their project work. This work was later turned into a product thanks to the following contributors:
[Yetunde Dada](https://github.com/yetudada), [Wesley Leong](https://www.linkedin.com/in/wesleyleong/), [Steve Ler](https://www.linkedin.com/in/song-lim-steve-ler-380366106/), [Viktoriia Oliinyk](https://www.linkedin.com/in/victoria-oleynik/), [Roxana Pamfil](https://www.linkedin.com/in/roxana-pamfil-1192053b/), [Nisara Sriwattanaworachai](https://www.linkedin.com/in/nisara-sriwattanaworachai-795b357/), [Nikolaos Tsaousis](https://www.linkedin.com/in/ntsaousis/), [Angel Droth](https://www.linkedin.com/in/angeldroth/), [Zain Patel](https://www.linkedin.com/in/zain-patel/), [Richard Oentaryo](https://www.linkedin.com/in/oentaryo/),
[Shuhei Ishida](https://www.linkedin.com/in/shuhei-i/), and [Francesca
Sogaro](https://www.linkedin.com/in/francesca-sogaro/).

CausalNex would also not be possible without the generous sharing from leading researches in the field of causal inference and we are grateful to everyone who advised and supported us, filed issues or helped resolve them, asked and answered questions or simply be part of inspiring discussions.
