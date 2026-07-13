# Release procedure

How to cut a new release of `dtai-veritas` to PyPI.

1. Bump `__version__` in `src/python/veritas/__init__.py`, commit to `main`.
2. Fast-forward the `deploy` branch to `main` and push it:
   ```sh
   git push origin main:deploy
   ```
   This triggers `.github/workflows/build_wheels.yml`'s `push: branches:
   [deploy]` condition, which runs the full `build_wheels` + `build_sdist`
   matrix as a **dry run** — it does not publish anything (the `upload_pypi`
   job only runs on a `release` event).
3. **Wait for that run to go fully green** before continuing
   ([Actions tab](https://github.com/laudv/veritas/actions/workflows/build_wheels.yml)).
   Do not proceed to step 4 on a red or in-progress run.
4. Create a git tag `vX.Y.Z` on the release commit and push it.
5. Create a new GitHub Release. Create a new git tag `vX.Y.Z` on the release
   commit. This fires the `release: published` event, which re-runs the same
   build matrix and — since it now succeeds/succeeded in step 3 with the same
   commit — runs `upload_pypi`, which publishes to PyPI via trusted publishing
   (OIDC, no token needed).

If the release-triggered run fails (e.g. a build job fails), nothing is
published: `upload_pypi` only runs if `needs: [build_wheels, build_sdist]` both
succeed. In that case it's safe to delete the tag and GitHub Release and redo
them once fixed, since PyPI never saw that version. This is *not* true anymore
once `upload_pypi` has actually started, though — PyPI does not allow
re-uploading a version's files even after deleting or yanking a release, so a
partial publish (e.g. some but not all wheels uploaded) permanently burns that
version number and forces a bump instead.
