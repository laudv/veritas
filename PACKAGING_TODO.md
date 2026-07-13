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
      README/README_TEMPLATE developer-install section rewritten; also fixed a bug
      surfaced along the way: `tests/readme_code.py`'s `make_moons()` had no fixed seed
      (non-deterministic README example output). (`editable.verbose` toggle was tried
      to silence the rebuild hook's stdout noise but is currently commented out.)
- [x] 2. Adopt PEP 621: move metadata into `pyproject.toml`'s `[project]` table; stop
      scraping version/author/etc. out of `__init__.py` via regex+exec. Use
      scikit-build-core's dynamic-version support (or setuptools-scm) so version has
      one source of truth. DONE 2026-07-13: `[project]` table added; version stays
      read from `__init__.py` but via scikit-build-core's regex metadata provider
      (single source of truth, no more manual exec in setup.py, which is now deleted).
- [x] 3. Declare optional extras in `pyproject.toml` (`milp`, `smt`, `xgboost`,
      `lightgbm`, `sklearn`) instead of undocumented try/except imports in `__init__.py`.
      DONE 2026-07-13: added `[project.optional-dependencies]`.
- [x] 4. Bump `requires-python` to >=3.9 or >=3.10 (3.8 is EOL); reconsider whether
      PyPy wheels are worth keeping in the build matrix. DONE 2026-07-13:
      `requires-python = ">=3.11"` (3.9 is already past EOL as of today, 3.10 has
      ~3 months left; 3.11 gives ~1 year more runway); dropped PyPy wheels from
      `cibuildwheel` (`skip = "pp* ..."`) per Laurens's call — pybind11/PyPy is a
      niche combo unlikely to have real users of this package; also skip
      cp38/cp39/cp310 in the cibuildwheel matrix to match the new floor.
- [x] 5. Add cp313 (and consider free-threaded) wheels to the `cibuildwheel` matrix;
      bump `cibuildwheel` itself from 2.17.0 to current. DONE 2026-07-13: bumped
      `pypa/cibuildwheel` action to v4.1.0 (2.17.0 -> 3.0.0 -> 4.0.0 -> 4.1.0
      changelog checked, no breaking impact here). cp313/cp314 now build
      automatically; cp314t (free-threaded) explicitly skipped (`*t-*`) since the
      vendored pybind11 (v2.12) predates free-threading support — revisit once
      item 6 updates it. Verified end-to-end: built and smoke-tested an actual
      manylinux wheel locally via `cibuildwheel --output-dir ...` + a clean venv.
- [x] 6. Update vendored submodules (pybind11 ~1.5yr stale, nlohmann_json v3.11.3);
      evaluate pybind11 stable-ABI (abi3) build to shrink the wheel matrix.
      DONE 2026-07-13: bumped `lib/pybind11` v2.12(dev)->v3.0.4 and
      `lib/nlohmann_json` v3.11.3->v3.12.0. pybind11 v3.0 changes ABI (rebuild
      required, no source changes needed for our bindings) and needs
      `PYBIND11_FINDPYTHON` for the modern `FindPython` CMake module, but our
      CMakeLists already finds `Python3` directly (not via pybind11's own
      discovery) so no CMakeLists change was needed. Verified: local editable
      build, full C++ `ctest` suite (9/9 pass), Python `unittest` suite, and an
      actual `cibuildwheel`-built manylinux wheel smoke-tested in a clean venv,
      all before and after the bump. Found a pre-existing (not a regression)
      bug along the way: `AddTree.__eq__` returns False after a JSON
      round-trip (`at.to_json()` -> `AddTree.from_json()`) even for a trivial
      one-tree ensemble — reproduces identically on the pre-bump submodule
      commits too. Flagged to Laurens, not fixed here (out of scope for
      packaging work).
      abi3/stable-ABI: **not possible** — pybind11 (checked v3.0.4 source)
      explicitly does not support `Py_LIMITED_API`
      (`include/pybind11/detail/internals.h`: "we cannot use Py_LIMITED_API
      anyway"). Dropping this sub-item; one wheel per CPython minor version
      remains necessary.
- [x] 7. Actually exercise built wheels in CI: fill in `test-requires`/`test-command`
      in `pyproject.toml`'s `[tool.cibuildwheel]` (currently commented out).
      DONE 2026-07-13: `test-requires = "xgboost lightgbm scikit-learn z3-solver
      gurobipy imageio pandas"` (matches every unconditional import across
      `tests/*.py`, checked file by file); `test-command = "cd {project} &&
      python -m unittest discover tests"` (`cd` is required — several tests
      open data/model files via paths relative to the repo root, and
      cibuildwheel does not run test-command from the project directory).
      Running this for real via `cibuildwheel --output-dir ...` surfaced a
      genuine bug that had never actually been exercised before: every
      `test_xgb_*` test in `test_converters.py` errored, because that whole
      test module imports `imageio` unconditionally at the top — so in every
      environment used so far (which lacked `imageio`), the module failed to
      import and these tests silently never ran. With `imageio` now installed,
      they revealed that `xgb_converter.py`'s `base_score` parsing
      (`float(...)`) breaks on newer XGBoost JSON exports, which encode it as
      a stringified array (e.g. `"[4.89E-1]"`) instead of a bare float. Fixed
      in `src/python/veritas/xgb_converter.py` (try `float()`, fall back to
      `json.loads()` for the array form). Verified: full local `unittest`
      suite (41/41 pass) and a full `cibuildwheel` build+test run, both green.

- [x] 12. (discovered during the first real `build_wheels.yml` dry run, v0.3.0 tag,
      2026-07-13) macOS wheel builds failed: `.github/workflows/build_wheels.yml`'s
      matrix used `macos-13`/`macos-14`, which are **deprecated** GitHub-hosted
      runner images (GitHub only supports the latest 2 macOS versions; currently
      15 and 26). This surfaced as a `z3-solver` install failure during the
      cibuildwheel test step on `macos-14`, because `z3-solver` publishes
      `macosx_15_0_arm64` wheels as of 4.15.5.0 (bumped from `macosx_13_0_arm64`)
      — the deprecated macOS-14 runner is below that floor. Fixed by updating the
      matrix to current runners (`macos-15-intel`, `macos-15`) rather than pinning
      `z3-solver` down — this fixes the immediate issue *and* gets CI off
      soon-to-be-removed runner images. No `pyproject.toml` change needed.

## Nice-to-have

- [ ] 8. Ship `py.typed` + generated `.pyi` stubs for `veritas_core` via
      `pybind11-stubgen`.
- [ ] 9. Merge `python_package.yml` and `build_wheels.yml` into one matrixed workflow.
- [ ] 10. Fill in `author_email` (currently `""` in `__init__.py`).
- [ ] 11. Reconsider vendoring pybind11/nlohmann_json as git submodules vs. depending
      on them as regular build requirements.
