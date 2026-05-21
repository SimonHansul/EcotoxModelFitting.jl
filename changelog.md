
## Changelog

### v0.1.9

- bugfix in `generate_loss_function`

### v0.1.10

- added `define_objective_function` for easier integration with Optim.jl

### v0.1.11

- added `quantitative_evaluation` function
- added `assign!` function to assign values from one `ComponentArray` to another

### v0.1.12

- bugfix in IO handling (`savedir`/`savetag` were not applied correctly)

### v0.2.0

- Introduced `Dataset` interface
    - defined a `target` function to compute the target from `data` and `sim`
    - implemented a test for use of Dataset interface + local optim for the minimal DEB case (fit to growth only)
- `PMCBackend`interface is still available for the time being, but will be depracated and removed

### v0.2.1

- added function `to_cvec` to convert `Parameters` object to `ComponentVector`

### v0.2.2

- added function `parameter_table` to print parameters as markdown table

### v0.2.3

- added support for `Evolutionary.CMAES`

### v0.2.4

- added weights to `Dataset`. weights can be supplied as a `DataFrame` column

### v0.2.5

- added kwarg `free_only` to `parameter_table`


### v0.2.6

- added CSV dependency

### v0.2.7

- changes to IO/utilities

### v0.3.0

- updated PMC backend to be compatible with the updated API
- what used to be `PMCBackend` is now `PMCBackend`