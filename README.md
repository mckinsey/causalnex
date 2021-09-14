![CausalNex](https://raw.githubusercontent.com/quantumblacklabs/causalnex/master/docs/source/causalnex_banner.png)

-----------------

| Theme | Status |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Latest Release | [![PyPI version](https://badge.fury.io/py/causalnex.svg)](https://pypi.org/project/causalnex/) |
| Python Version | [![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://pypi.org/project/causalnex/) |
| `master` Branch Build | [![CircleCI](https://circleci.com/gh/quantumblacklabs/causalnex/tree/master.svg?style=shield&circle-token=92ab70f03f3183655473dad16be641959cd31b83)](https://circleci.com/gh/quantumblacklabs/causalnex/tree/master) |
| `develop` Branch Build | [![CircleCI](https://circleci.com/gh/quantumblacklabs/causalnex/tree/develop.svg?style=shield&circle-token=92ab70f03f3183655473dad16be641959cd31b83)](https://circleci.com/gh/quantumblacklabs/causalnex/tree/develop) |
| Documentation Build | [![Documentation](https://readthedocs.org/projects/causalnex/badge/?version=latest)](https://causalnex.readthedocs.io/) |
| License | [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) |
| Code Style | [![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black) |


## What is CausalNex?

> "A toolkit for causal reasoning with Bayesian Networks."

CausalNex aims to become one of the leading libraries for causal reasoning and "what-if" analysis using Bayesian Networks. It helps to simplify the steps:
 - To learn causal structures,
 - To allow domain experts to augment the relationships,
 - To estimate the effects of potential interventions using data.

## Why CausalNex?

CausalNex is built on our collective experience to leverage Bayesian Networks to identify causal relationships in data so that we can develop the right interventions from analytics. We developed CausalNex because:

- We believe **leveraging Bayesian Networks** is more intuitive to describe causality compared to traditional machine learning methodology that are built on pattern recognition and correlation analysis.
- Causal relationships are more accurate if we can easily **encode or augment domain expertise** in the graph model.
- We can then use the graph model to **assess the impact** from changes to underlying features, i.e. counterfactual analysis, and **identify the right intervention**.

In our experience, a data scientist generally has to use at least 3-4 different open-source libraries before arriving at the final step of finding the right intervention.  CausalNex aims to simplify this end-to-end process for causality and counterfactual analysis.

## What are the main features of CausalNex?

The main features of this library are:

- Use state-of-the-art structure learning methods to understand conditional dependencies between variables
- Allow domain knowledge to augment model relationship
- Build predictive models based on structural relationships
- Fit probability distribution of the Bayesian Networks
- Evaluate model quality with standard statistical checks
- Simplify how causality is understood in Bayesian Networks through visualisation
- Analyse the impact of interventions using Do-calculus

## How do I install CausalNex?

CausalNex is a Python package. To install it, simply run:

```bash
pip install causalnex
```

Since pygraphviz can be difficult to install, esp. on Windows machines, the requirement is optional.
If you want to use the causalnex native plotting tools, you can use
```bash
pip install "causalnex[plot]"
```
Alternatively, you can use the `networkx` drawing functionality for visualisations with fewer dependencies.

Use `all` for a full installation of dependencies (only the plotting right now):
```bash
pip install "causalnex[all]"
```

See more detailed installation instructions, including how to setup Python virtual environments, in our [installation guide](https://causalnex.readthedocs.io/en/latest/02_getting_started/02_install.html) and get started with our [tutorial](https://causalnex.readthedocs.io/en/latest/03_tutorial/01_first_tutorial.html).

## How do I use CausalNex?

You can find the documentation for the latest stable release [here](https://causalnex.readthedocs.io/en/latest/). It explains:

- An end-to-end [tutorial on how to use CausalNex](https://causalnex.readthedocs.io/en/latest/03_tutorial/01_first_tutorial.html)
- The [main concepts and methods](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html) in using Bayesian Networks for Causal Inference

> Note: You can find the notebook and markdown files used to build the docs in [`docs/source`](docs/source).

## Can I contribute?

Yes! We'd love you to join us and help us build CausalNex. Check out our [contributing](CONTRIBUTING.md) documentation.

## How do I upgrade CausalNex?

We use [SemVer](http://semver.org/) for versioning. The best way to upgrade safely is to check our [release notes](RELEASE.md) for any notable breaking changes.

## What licence do you use?

See our [LICENSE](LICENSE.md) for more detail.

## We're hiring!

Do you want to be part of the team that builds CausalNex and [other great products](https://quantumblack.com/labs) at QuantumBlack? If so, you're in luck! QuantumBlack is currently hiring Machine Learning Engineers who love using data to drive their decisions. Take a look at [our open positions](https://www.quantumblack.com/careers/current-openings#content) and see if you're a fit.
