# R CMD check entry point. Run directly with:
#   Rscript -e 'testthat::test_dir("rust/scfast-r/tests/testthat", package = "scfastR", load_package = "installed")'
library(testthat)
library(scfastR)

test_check("scfastR")
