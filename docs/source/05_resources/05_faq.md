# Frequently asked questions

> *Note:* This documentation is based on `CausalNex 0.5.0`, if you spot anything that is incorrect then please create an [issue](https://github.com/quantumblacklabs/causalnex/issues) or pull request.

## What is CausalNex?

[CausalNex](https://github.com/quantumblacklabs/causalnex) is a python library that allows data scientists and domain experts to co-develop models which go beyond correlation to consider causal relationships. It was originally designed by [Paul Beaumont](https://www.linkedin.com/in/pbeaumont/) and [Ben Horsburgh](https://www.linkedin.com/in/benhorsburgh/) to solve challenges they faced in inferencing causality in their project work.

This work was later turned into a product thanks to the following contributors: [Ivan Danov](https://github.com/idanov), [Dmitrii Deriabin](https://github.com/DmitryDeryabin), [Yetunde Dada](https://github.com/yetudada), [Wesley Leong](https://www.linkedin.com/in/wesleyleong/), [Steve Ler](https://www.linkedin.com/in/song-lim-steve-ler-380366106/), [Viktoriia Oliinyk](https://www.linkedin.com/in/victoria-oleynik/), [Roxana Pamfil](https://www.linkedin.com/in/roxana-pamfil-1192053b/), [Fabian Peters](https://www.linkedin.com/in/fabian-peters-6291ab105/), [Nisara Sriwattanaworachai](https://www.linkedin.com/in/nisara-sriwattanaworachai-795b357/) and [Nikolaos Tsaousis](https://www.linkedin.com/in/ntsaousis/).

## What are the benefits of using CausalNex?

It is important to consider the primary benefits of CausalNex in the context of an end-to-end causality and counterfactual analysis.

As we see it, CausalNex:

- **Generates transparency and trust in models** it creates by allowing users to collaborate with domain experts during the modelling process.
- Uses an **optimised structure learning algorithm**, [NOTEARS](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf) where the runtime to learn structure is no longer exponential but scales cubically with number of nodes.
- **Enables adding known relationships/removing spurious correlations** so that your model can better consider causal relationships in data.
- **Visualises networks using common tools** built upon [NetworkX](https://networkx.github.io/), allowing users to understand relationships in their data more intuitively, and work with experts to encode their knowledge.
- **Streamlines the use of Bayesian Networks** for an end-to-end counterfactual analysis, which in the past was a complicated process involving the use of at least three separate open source libraries, each with its own interface.

## When should you consider using CausalNex?

CausalNex is created specifically for data scientists who would like an efficient and intuitive process to identify causal relationships and the right intervention through data and collaboration with domain experts.

## Why NOTEARS algorithm over other structure learning methods?

Historically, structure learning has been a very **hard** problem. We are interested in looking for the optimal directed acyclic graph (DAGs) that describes the conditional dependencies between variables. However, the search space for this is **combinatorial** and scales **super-exponentially** with the number of nodes. [NOTEARS](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf) algorithm cleverly introduces a new optimisation heuristic and approach to solving this problem, where the runtime for this is no longer exponential but scales **cubically** with the number of nodes.

## What is the recommended type of dataset to be used in NOTEARS?

[NOTEARS](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf) works by detecting if a small increase in the value of the node will result in an increase in another node. If there is, NOTEARS will be able to capture this and assert that this is a causal relationship. Therefore, we highly recommend that the dataset to be used is **continuous**.

**Categorical variables** like blood type **won’t be able to work** in this case. Nonetheless, after learning the structure using NOTEARS, one can still manually add the relationships for these features to the structure based on their domain knowledge.

## What is the recommended number of samples for satisfactory performance?

According to the benchmarking done on **synthetic dataset** in-house, it is highly recommended that **at least 1000 samples** is used to get a satisfactory performance. We also discovered that any further increase than 1000 samples **does not help improve the accuracy** regardless of number of nodes, and it takes a **longer time** to run.

## Why can my StructureModel be cyclic, but not my BayesianNetwork?

StructureModel is used when discovering the causal structure of a dataset. Part of this process is adding, removing, and flipping edges, until the appropriate structure is completed. As edges are modified, cycles can be temporarily introduced into the structure, which would raise an Exception within a BayesianNetwork, which is a specialised **directed acyclic graph**.

Once the structure is finalised, and is acyclic, then it can be used to create a [BayesianNetwork](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html).


## Why a separate data pre-processing process for probability fitting than structure learning? / Why discretise data in probability fitting?

We treat Bayesian Network probability fitting and Structure Learning as two separate problems. The data for Structure Learning should be continuous for the causal relationships to be learnt. **Once we already knew the causal relationship between all the nodes**, we can start doing probability fitting. At the moment, we are **only supporting discrete Bayesian Network model**, and this requires the continuous features to be discretised.

## Why call fit_node_states before fit_cpds?

Before fitting, the model first has to know how many states each node has to carry out the computations. Alternatively, one can also call **fit_node_states_and_cpds**. However, there is a chance that this might not work if one were to do train/test splitting as the model might not see all the possible states.

For example, rare blood type like AB-negative might not appear in the training data but in the test data. Therefore, we strongly encourage users to do **fit_node_states using all data** and **fit_cpds using training data** to test the model quality, so that the model knows all the possible states that each node can have.

## What is Do-intervention and when to use it?

[Do-intervention](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html) is symbolically described as p(y|do(x)). It asks the question of what is the probability distribution of Y if we were to **set** the value of X to x **arbitrarily**.

For example, we have 50% of males and 50% of females in the world, but we might be interested to learn about the probability distribution of happiness index if we had 80% of females and 20% males in the world.

Do-intervention is very useful in **counterfactual analysis**, where we are interested to know if the outcomes would have been different if we had taken a different action/intervention.

## How can I make inference faster?

At the moment, the algorithm calculates the probability of **every node** in a Bayesian Network. If users are interested in making inference of the target node faster, user can remove nodes that are independent from the target node, and also children of the target node. For example, if we have C<-A->B->D and we want to learn P(B|A), we can remove C and D to make the inference faster.

## How does CausalNex compare to other projects, e.g. CausalML, DoWhy?

The following points describe how we are unique comparing to the others:
1) We are one of the very few causal packages that use **Bayesian Networks** to model the problems. Most of the causal packages use statistical matching technique like **propensity score matching** to approach these problems.
2) One of the main hurdles to applying Bayesian Networks is to find the optimal graph structure. In CausalNex, We **simplify** this process by providing the ability for the users to learn the graph structure through: i) **encoding domain expertise** by manually adding the edges, and ii) **leveraging the data** using the state-of-the-art [structure learning algorithm](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf).
3) We provide the ability for the users to do **counterfactual analysis** using Bayesian Network by introducing **Do-Calculus**, which is not commonly found in Bayesian Network packages.

## What version of Python does CausalNex use?

CausalNex is built for Python 3.6, 3.7, and 3.8.

## How do I upgrade CausalNex?

[release]: https://tinyurl.com/f7jw6cwz

We use [SemVer](http://semver.org/) for versioning. The best way to upgrade safely is to check our [release notes][release] for any notable breaking changes.

Once CausalNex is installed, you can check your version as follows:

```
pip show causalnex
```

To later upgrade CausalNex to a different version, simply run:

```
pip install causalnex -U
```

## How can I find out more CausalNex?

CausalNex is on GitHub, and our preferred community channel for feedback is through [GitHub issues](https://github.com/quantumblacklabs/causalnex/issues). You can find news about updates and new features introduced by heading over to [RELEASE.md][release].

## Where can I learn more about Bayesian Networks?

You can read our [documentation](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html) to know more about the concepts and other useful references with regards to using Bayesian Networks for Causal Inference.
