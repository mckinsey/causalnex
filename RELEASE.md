# Upcoming release


# Release 0.7.0

* Added plotting tutorial to the documentation
* Updated `viz.draw` syntax in tutorial notebooks
* Bugfix on notears lasso (`from_numpy_lasso` and `from_pandas_lasso`) where the non-negativity constraint was not being set
* Added DAG-based synthetic data generator for mixed types (binary, categorical, continuous) using a linear SEM approach.
* Unpinned some requirements

# Release 0.6.0

* support for newer versions of scikit-learn
* classification report now returns dict in line with scikit-learn
* adding dynotears (`from_numpy_dynamic`, an algorithm for structure learning on Dynamic Bayesian Networks)
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
[Yetunde Dada](https://github.com/yetudada), [Wesley Leong](https://www.linkedin.com/in/wesleyleong/), [Steve Ler](https://www.linkedin.com/in/song-lim-steve-ler-380366106/), [Viktoriia Oliinyk](https://www.linkedin.com/in/victoria-oleynik/), [Roxana Pamfil](https://www.linkedin.com/in/roxana-pamfil-1192053b/), [Nisara Sriwattanaworachai](https://www.linkedin.com/in/nisara-sriwattanaworachai-795b357/) and [Nikolaos Tsaousis](https://www.linkedin.com/in/ntsaousis/).

CausalNex would also not be possible without the generous sharing from leading researches in the field of causal inference and we are grateful to everyone who advised and supported us, filed issues or helped resolve them, asked and answered questions or simply be part of inspiring discussions.
