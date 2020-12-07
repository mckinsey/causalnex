# Introduction


CausalNex is a Python library that uses Bayesian Networks to combine machine learning and domain expertise for causal reasoning.
You can use CausalNex to uncover structural relationships in your data, learn complex distributions,
and observe the effect of potential interventions.

## Main features of CausalNex

The CausalNex library has the following features:

- Deploys state-of-the-art structure learning method, [DAG with NO TEARS](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf), to understand conditional dependencies between variables
- Allows domain knowledge to augment model relationships
- Builds predictive models based on structural relationships
- Understands model probability
- Evaluates model quality with standard statistical checks
- Visualisation which simplifies how causality is understood
- Analyses the impact of interventions using Do-calculus

## Learning About CausalNex

In the next few chapters, you will learn how to install and set up CausalNex, and how to use it on your own projects.
Once you are set up, to get a feel for CausalNex, we suggest working through our example tutorial project.
Advanced users looking for in-depth information should consult the User Guide.
You can also check out the resources section for answers to frequently asked questions and the API reference documentation for further, detailed information.

## Assumptions

We have designed the documentation in general, and the tutorial in particular, for beginners to get started using Bayesian Networks on their projects. If you an have elementary knowledge of Python and Bayesian Networks then you may find the CausalNex learning curve more challenging. However, we have simplified the tutorial by providing all the Python functions necessary to create your first CausalNex project.

Note: There are a number of excellent online resources for learning Python, but be aware that
you should choose those that reference Python 3, as CausalNex is built for Python 3.6+.
There are many curated lists of online resources, such as:

- [Official Python programming language website](https://www.python.org/)
- [List of free programming books and tutorials](https://github.com/EbookFoundation/free-programming-books/blob/master/free-programming-books.md#python)

There are also several excellent online resources for learning about Bayesian Networks, such as:

- [Lecture notes](https://ermongroup.github.io/cs228-notes/) on Probabilistic graphical models based on Stanford CS228;
- [An Introduction to Bayesian Network Theory and Usage](http://infoscience.epfl.ch/record/82584) by T. Stephenson;
- [PGMPY tutorial](https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb).
