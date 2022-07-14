FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt install -y python3.8 python3-pip libgraphviz-dev graphviz

RUN ln -s $(which python3) /usr/local/bin/python

# install requirements
RUN python3 -m pip --no-cache-dir install \
    'networkx~=2.5' \
    'numpy>=1.14.2, <2.0' \
    'pandas>=1.0, <2.0' \
    'pathos>=0.2.7, <0.3.0' \
    'pgmpy>=0.1.12, <0.2.0' \
    'scikit-learn>=0.22.0, <0.25.0, !=0.22.2.post1, !=0.24.1' \
    'scipy>=1.2.0, <1.7' \
    'torch>=1.7, <2.0' \
    'wrapt>=1.11.0, <1.13'

# install test requirements
RUN python3 -m pip --no-cache-dir install \
    'Cython>=0.29, <1.0' \
    'flake8>=3.5, <4.0' \
    'ipython>=7.0, <7.17' \
    'isort>=4.3.16, <5.0' \
    'matplotlib~=3.3' \
    'mdlp-discretization~=0.3.3' \
    'mock>=2.0.0, <3.0' \
    'pre-commit>=2.9.2' \
    'pygraphviz>=1.5, <2.0' \
    'pylint>=2.7.2, <3.0' \
    'pytest-cov>=2.5, <3.0' \
    'pytest-mock>=1.7.1,<2.0' \
    'pytest>=4.3.0,<5.0' \
    'scikit-learn>=0.24.2'

# install doc requirements
RUN python3 -m pip --no-cache-dir install \
    'click>=7.0, <8.0' \
    'ipykernel>=4.8.1, <5.0' \
    'ipython_genutils>=0.2.0' \
    'jinja2>=2.3, <3.0' \
    'jupyter_client>=5.1, <7.0' \
    'Markupsafe<2.1' \
    'nbconvert>=5.0, <6.0' \
    'nbsphinx==0.4.2' \
    'nbstripout==0.3.3' \
    'patchy>=1.5, <2.0' \
    'pydot>=1.4, <2.0' \
    'pygments>=2.6.1, <3.0' \
    'pygraphviz>=1.5, <2.0' \
    'recommonmark>=0.5.0, <1.0' \
    'sphinx-autodoc-typehints>=1.6.0, <2.0' \
    'sphinx-markdown-tables>=0.0.15, <1.0' \
    'sphinx>=3.0.4, <4.0' \
    'sphinx_copybutton>=0.2.5, <1.0' \
    'sphinx_rtd_theme>=0.4.3, <1.0'