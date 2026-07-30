# Releasing

S2Mosaic is published to PyPI by tag-push. The version comes from the git tag
itself (via `setuptools-scm`) — there is **no version file to bump**.

## Versioning

Follow [SemVer](https://semver.org/):

- **Major** (`v2.0.0` → `v3.0.0`): incompatible API change — removed args,
  changed signatures, behavioural changes that break existing callers.
- **Minor** (`v2.0.0` → `v2.1.0`): backwards-compatible feature added.
- **Patch** (`v2.0.0` → `v2.0.1`): backwards-compatible bug fix.
- **Pre-release** (`v2.0.0b1`, `v2.0.0b2`, …): a beta of the next version. PyPI
  treats these as pre-releases, so `pip install s2mosaic` keeps resolving to the
  latest stable and only `pip install --pre s2mosaic` picks them up. Use one when
  a change needs real-world exposure before it becomes the default install.

Any commit that lands on `main` between releases shows up at install time as
a development version like `2.0.1.dev3+g1234abc` — useful but not something
you publish.

## Release checklist

1. **Sync `main`** locally:
   ```bash
   git checkout main
   git pull
   ```

2. **Update [`CHANGELOG.md`](CHANGELOG.md)**: add a `## [<version>] - <YYYY-MM-DD>`
   section at the top, above the previous release, with the changes grouped under
   `### Added` / `### Changed` / `### Fixed`. There is no `[Unreleased]` block —
   entries are written straight under the version being released. Commit on `main`:
   ```bash
   git commit -am "Cut <version>: <short summary>"
   git push
   ```

3. **Tag and push**:
   ```bash
   git tag v<version>          # e.g. v2.0.0
   git push origin v<version>
   ```
   The tag must match `v<major>.<minor>.<patch>`, optionally with a pre-release
   suffix — the publish workflow filters on `v[0-9]+.[0-9]+.[0-9]+*`, so
   `v2.0.0b3` and `v2.0.0rc1` both match. Pre-releases are what `v2.0.0b1` and
   `v2.0.0b2` used.

4. **Approve the deploy**: the [`Publish to PyPI`](.github/workflows/publish.yml)
   workflow runs `uv build` and uploads to PyPI via OIDC trusted publishing.
   The `pypi` GitHub environment is gated by a required reviewer — go to
   **Actions → Publish to PyPI → Review deployments → Approve** to release.

5. **Verify**:
   - PyPI page: <https://pypi.org/project/s2mosaic/>
   - GitHub Releases: <https://github.com/DPIRD-DMA/S2Mosaic/releases>
     (auto-generated release notes + wheel/sdist attached).
