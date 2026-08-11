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
- `ContTempNetwork` / `ContTempInstNetwork`: removed the constructor parameters
  `relabel_nodes`, `reset_event_table_index`, and `node_to_label_dict`. Data
  normalization is controlled by the new `sanitize_data` parameter (default
  `True`); the fast path is `sanitize_data=False`, which uses the input as-is
  and emits a `UserWarning`.
- `ContTempInstNetwork` now represents pulses with
  `ending_times == starting_times`; supplied `ending_times` are ignored with a
  `UserWarning`.
- With `sanitize_data=True`, event tables are normalized on construction
  (contiguous 0..N-1 node ids, chronological sort by
  `(starting_times, ending_times)`, and zero-based `RangeIndex`). The
  `sanitize_data=False` fast path preserves the input as-is.

### Added
- ASV benchmark suite (`benchmarks/`)
- `compute_static_adjacency_matrix()` supports binary edge-presence output by
  default and optional weighted duration or event-count aggregation via
  `weighted=True`.
- Instantaneous networks support weighted event counts, while interval
  networks support weighted durations and event counts.
- Pulse activity windows use the half-open interval `[t_start, t_end)`; a
  default `end_time=None` includes pulses at the final network time.
- `ContTempInstNetwork` laplacian computation uses an internal terminal
  boundary so the final pulse is processed without changing public time
  metadata.
- Added transition-matrix preparation helpers in `tempnet.utils`:
  `_prepare_inter_transition_matrix()` and
  `_threshold_and_row_normalize()`.
- Added `num_active_events`, `num_active_nodes` as methods of `ContTempNetwork` class in the `temporal_network.py`. 
  They compute the number of active edges and nodes within a range of the t_start and t_end. 
- New `tempnet.sanitize` module with public helpers `sanitize_events_table()`
  (copy by default, or `inplace=True`) and `needs_sanitization()`, both
  exported from `tempnet`.
- `pytest --run-network` opt-in flag; Zenodo-dependent tests (marker
  `network`) are skipped by default.

### Fixed
- Instantaneous-network static adjacency no longer interprets zero-duration
  pulses as duration-one events; weighted pulse adjacency counts selected
  events explicitly.
- The final instantaneous pulse is no longer omitted from laplacian
  computation.
- `num_active_edges()` now uses the network-specific active-event selector.
- `compute_transition_matrices()` no longer mutates the stored `inter_T`
  matrices while preparing or normalizing them.
- Removed duplicate `active_nodes()` and `num_active_nodes()` definitions so
  both methods use the shared active-event selection logic.
- Constructor invariants are now enforced for all input paths: unsorted
  list/DataFrame inputs are sorted with a reset index; duplicate or named
  DataFrame indices no longer corrupt the time grid / Laplacian computation.
- Windowed `compute_inter_transition_matrices` (after
  `compute_laplacian_matrices(t_start=..., t_stop=...)`) now pairs each
  Laplacian with the inter-event time of its own step; previously taus were
  counted from `times[0]`, producing wrong transition matrices for any window
  with `_k_start_laplacians > 0`.
- Invalid methods passed directly to `_compute_single_T` now raise
  `ValueError` instead of failing with `UnboundLocalError`.
- Invalid `mfp_exp(non_norm=...)` values now raise `ValueError` instead of
  failing with `UnboundLocalError`.
- The Zenodo mice dataset is downloaded once per test session (session-scoped
  fixture) instead of once per test.
