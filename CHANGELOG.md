# Changelog

## Unreleased

### Breaking Changes
- Removed the linear-approximation transition-matrix API from `ContTempNetwork`:
  `compute_lin_inter_transition_matrices()` and `compute_lin_transition_matrices()`.
- Removed the corresponding linear-approximation persistence helpers:
  `save_inter_T_lin()` and `save_T_lin()`.
- Removed module-level linear-approximation helpers from `tempnet.temporal_network`:
  `lin_approx_trans_matrix()`, `sparse_lin_approx()`,
  `compute_stationary_transition()`, and `sparse_stationary_trans()`.
- Removed stationary-transition storage used only by the linear approximation:
  `_stationary_trans` is no longer computed, saved, or loaded by default.
- Removed default save/load support for linear-approximation matrix attributes:
  `inter_T_lin`, `T_lin`, and `delta_inter_T_lin` are no longer part of the
  default matrix list for `save()` and `load()`.

- Moved `csc_row_normalize`,  `numpy_rebuild_nnz_rowcol`, `set_to_zeroes`, `set_to_ones`, `find_spectral_gap`, ` remove_nnz_rowcol` from `temporal_network.py` to `utils.py`. 
- Change the name of `Tcsr` to just `T` in function `set_to_zeroes` in `utils.py`
- Moved `numpy_rebuild_nnz_rowcol` from `utils.py` to `stochmat` package.
- Added a new argument `dynamics` (`heat` and `rw`) for computing laplacians in `temporal_network.py`
  (function name: `compute_laplacian_matrices`)
- Moved `sparse_lapl_expm` and `compute_subspace_expm` from `temporal_network.py` to `faster_expm.py`.
- Renamed `parallel_expm.py` to `faster_expm.py`.
