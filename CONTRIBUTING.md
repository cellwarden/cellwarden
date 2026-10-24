# Contributing to CellWarden

Thank you for testing and improving CellWarden.

## Reporting what you found
- **Tried it on your data?** Open an issue with the *"I tested CellWarden"* form: dataset size,
  what ran, how long it took, what broke, and how useful the review agents were.
- **Bugs:** include `cellwarden doctor` output, the command, and the last lines of the failing
  stage's log (`<run>/logs/<stage>.log`). Remove anything confidential from logs first.
- **Questions and ideas:** use GitHub Discussions.

## Changing the code
1. Fork, create a branch, and keep each pull request to one change.
2. Run the tests that cover your change: `python -m pytest tests/python`, the R tests with
   `Rscript -e 'testthat::test_dir("tests/R")'`, and for pipeline changes
   `tests/integration/run_full.sh` (see `tests/README.md`).
3. Write the code like the code around it, and update the docs it changes.

## Sign-off (Developer Certificate of Origin)
Every commit must be signed off, confirming you wrote it or have the right to submit it under
the project's licence (Apache-2.0), as in the Developer Certificate of Origin
(https://developercertificate.org):

    git commit -s -m "what you changed"

This adds `Signed-off-by: Your Name <you@example.org>`. Contributions are accepted under
Apache-2.0; the copyright of what you write stays yours.
