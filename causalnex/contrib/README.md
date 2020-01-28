# CausalNex contrib

The contrib directory is meant to contain user contributions, these
contributions might get merged into core CausalNex at some point in the future.

When create a new module in `contrib`, place it exactly where it would be if it
was merged into core CausalNex.

For example, functions to plot network diagrams are under the core package `causalnex.plotting`. If you are
contributing a new visualisation or plot you should have the following directory:
`causalnex/contrib/my_project/plotting/` - i.e., the name of your project before the
`causalnex` package path.

This is how a module would look like under `causalnex/contrib`:
```
causalnex/contrib/my_project/plotting/
    my_module.py
    README.md
```

You should put you test files in `tests/contrib/my_project`:
```
tests/contrib/my_project
    test_my_module.py
```

## Requirements

If your project has any requirements that are not in the core `requirements.txt`
file. Please add them in `setup.py` like so:
```
...
extras_require={
        'my_project': ['requirement1==1.0.1', 'requirement2==2.0.1'],
    },
```

Please notice that a readme with instructions about how to use your module
and 100% test coverage are required to accept a PR.
