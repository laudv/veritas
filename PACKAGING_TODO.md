# Python packaging modernization

Tracking list from the 2026-07-13 packaging review. Check items off as they land;
each should be its own commit (and ideally its own PR) so they can be reviewed/reverted
independently.

## Worth doing

- [x] 1. Migrate build backend from `scikit-build` to `scikit-build-core`; get a real
      PEP 660 editable install (`pip install -e .`) working, replacing the manual
      manual_build/ + symlink workflow in the README. DONE 2026-07-13: CMakeLists.txt
      install destination changed to `${PROJECT_NAME}` (relative, wheel-package-root
      style); `[tool.scikit-build.editable] rebuild = true` gives auto-rebuild-on-import;
      README/README_TEMPLATE developer-install section rewritten; also fixed two bugs
      surfaced along the way: `tests/readme_code.py`'s `make_moons()` had no fixed seed
      (non-deterministic README example output), and the rebuild hook's stdout noise is
      now silenced via `editable.verbose = false`.
- [x] 2. Adopt PEP 621: move metadata into `pyproject.toml`'s `[project]` table; stop
      scraping version/author/etc. out of `__init__.py` via regex+exec. Use
      scikit-build-core's dynamic-version support (or setuptools-scm) so version has
      one source of truth. DONE 2026-07-13: `[project]` table added; version stays
      read from `__init__.py` but via scikit-build-core's regex metadata provider
      (single source of truth, no more manual exec in setup.py, which is now deleted).
- [x] 3. Declare optional extras in `pyproject.toml` (`milp`, `smt`, `xgboost`,
      `lightgbm`, `sklearn`) instead of undocumented try/except imports in `__init__.py`.
      DONE 2026-07-13: added `[project.optional-dependencies]`.
- [ ] 4. Bump `requires-python` to >=3.9 or >=3.10 (3.8 is EOL); reconsider whether
      PyPy wheels are worth keeping in the build matrix.
- [ ] 5. Add cp313 (and consider free-threaded) wheels to the `cibuildwheel` matrix;
      bump `cibuildwheel` itself from 2.17.0 to current.
- [ ] 6. Update vendored submodules (pybind11 ~1.5yr stale, nlohmann_json v3.11.3);
      evaluate pybind11 stable-ABI (abi3) build to shrink the wheel matrix.
- [ ] 7. Actually exercise built wheels in CI: fill in `test-requires`/`test-command`
      in `pyproject.toml`'s `[tool.cibuildwheel]` (currently commented out).

## Nice-to-have

- [ ] 8. Ship `py.typed` + generated `.pyi` stubs for `veritas_core` via
      `pybind11-stubgen`.
- [ ] 9. Merge `python_package.yml` and `build_wheels.yml` into one matrixed workflow.
- [ ] 10. Fill in `author_email` (currently `""` in `__init__.py`).
- [ ] 11. Reconsider vendoring pybind11/nlohmann_json as git submodules vs. depending
      on them as regular build requirements.
