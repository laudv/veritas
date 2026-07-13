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
- [x] 13. (discovered on the next `deploy` dry run after item 12's fix, 2026-07-13)
      macOS build still failed, this time on `test_converters.py`'s `import
      xgboost`: `libxgboost.dylib` couldn't load `libomp.dylib` (OpenMP
      runtime) — xgboost's macOS wheels link it dynamically but don't bundle
      it, and it's not preinstalled on GitHub's macOS runners (this is
      XGBoost's own documented macOS caveat). Fixed with
      `[tool.cibuildwheel.macos] before-test = "brew install libomp"`.
      Also cleaned up two cibuildwheel-4.x no-op warnings surfaced in the same
      run: `pp*`/`cp38-*` skip selectors are gone (v3.0+ no longer builds
      PyPy or Python <3.9 by default, so those selectors matched nothing).
      Not verified end-to-end locally — no macOS runner available in this
      sandbox; verified only via TOML validity + cibuildwheel accepting the
      config. Needs a real `deploy` dry run to confirm.
- [x] 14. (discovered on the next `deploy` dry run, 2026-07-13) `windows-latest`
      failed with `ImportError: Start directory is not importable: 'tests'`.
      First attempt: added an empty `tests/__init__.py`, believing it was the
      standard `unittest discover` gotcha — **wrong diagnosis**, it recurred
      identically on a later run even with `tests/__init__.py` present. Real
      cause (confirmed against CPython's actual `unittest/loader.py` source):
      plain `cd` in `cmd.exe` does not switch drives, and GitHub's Windows
      runners put the checkout on `D:` while the shell's active drive when
      cibuildwheel runs `test-command` is elsewhere — so
      `test-command = "cd {project} && ..."`'s `cd` silently no-ops,
      `discover` runs from the wrong directory, "tests" isn't found as a
      directory there, and `unittest` falls back to importing it as a dotted
      module name, which fails with the exact same error text (unrelated to
      `__init__.py`, which is why that "fix" didn't help). Fixed with a
      `[tool.cibuildwheel.windows]` `test-command` override using `cd /d`
      instead of `cd` (`/d` also switches drives). `tests/__init__.py` is
      harmless and left in place (doesn't hurt, matches common convention)
      but wasn't actually the fix for this. Not verified end-to-end — no
      Windows runner in this sandbox; needs a real `deploy` dry run.
      Separately, `macos-15-intel` failed with
      `z3.z3types.Z3Exception: libz3.dylib not found.` raised from deep inside
      `z3-solver` itself at import time — since `veritas/__init__.py`'s
      optional-import guards intentionally only catch `ModuleNotFoundError`
      (by design, not a bug — see memory `feedback_exception_handling`), this
      uncaught exception broke `import veritas` entirely, cascading into ~40
      unrelated test failures. Decided not to chase the native-library issue
      further (SMT/z3 is an optional integration, and per Laurens real-world
      Windows usage of it is negligible too): restructured
      `[tool.cibuildwheel]`'s `test-requires` to exclude `z3-solver` by
      default (so Windows/macOS don't install it) and re-add it only via a
      `[tool.cibuildwheel.linux]` `test-requires` override — z3/SMT is now
      exercised in CI on Linux only. Made `test_verifier.py`/
      `test_z3backend.py` skip cleanly on `ModuleNotFoundError` (mirroring the
      existing `test_groot.py` pattern) rather than erroring when it's
      missing. Verified locally by simulating `z3` being
      genuinely absent: suite now reports `OK (skipped=2)`, exit code 0,
      instead of errors — and confirmed the real z3-available path still runs
      those tests normally (not accidentally always-skipped).
      Still open: whether to set `fail-fast: false` on the `build_wheels`
      matrix so unrelated platform failures don't cancel each other's jobs
      mid-debug (came up but wasn't decided).
- [x] 16. Set `fail-fast: false` on the `build_wheels` matrix strategy in
      `.github/workflows/build_wheels.yml` (resolves item 14's open question)
      — no cost concern since Actions minutes are free for this public repo,
      and it immediately paid off: the very next `deploy` run surfaced a
      genuinely new, unrelated Windows failure instead of hiding behind
      whatever failed first.
- [x] 17. (surfaced by item 16, 2026-07-13) `windows-latest` failed on
      `print(at[0])`: `UnicodeEncodeError: 'charmap' codec can't encode
      characters`. `tree.hpp`'s `print_node` uses box-drawing characters
      (`├─ └─ │`); Windows consoles default `stdout` to a legacy codepage
      (e.g. `cp1252`) that can't encode them. Fixed by setting
      `PYTHONUTF8 = "1"` in `[tool.cibuildwheel.windows]`'s `environment`,
      which forces Python's text I/O to UTF-8 regardless of console codepage
      (this is what PEP 540/686 eventually make the default anyway).
      Note: this only fixes cibuildwheel's CI test venv — a real end user on
      Windows running our code in a plain, non-UTF-8-configured console could
      still hit this same crash calling `print(tree)`/`print(addtree)`; a
      full fix would mean not relying on console-encoding for the C++ side's
      box-drawing output. Not fixed at that level here (out of scope for this
      CI debugging pass) — worth a follow-up if this comes up as a real user
      report. Not verified end-to-end — no Windows runner in this sandbox.

## Nice-to-have

- [ ] 8. Ship `py.typed` + generated `.pyi` stubs for `veritas_core` via
      `pybind11-stubgen`.
- [ ] 9. Merge `python_package.yml` and `build_wheels.yml` into one matrixed workflow.
- [ ] 10. Reconsider vendoring pybind11/nlohmann_json as git submodules vs. depending
      on them as regular build requirements.
