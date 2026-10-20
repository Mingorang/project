# Install every R package this project uses. Run once:
#     Rscript install_r_packages.R
# (or open R / RStudio and run this file with source("install_r_packages.R"))
#
#   ggplot2, png     -> analyze_looped_results.R   (ggplot2 draws the charts, png reads image_N.png)
#   dplyr, zoo       -> transform#01.r             (together with ggplot2)
# grid, grDevices, stats and utils ship with R itself, so there is nothing to install for them.

needed  <- c("ggplot2", "png", "dplyr", "zoo")
missing <- setdiff(needed, rownames(installed.packages()))

if (length(missing) > 0) {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All R packages are already installed.")
}

# Fail loudly if something did not install.
invisible(lapply(needed, function(p) library(p, character.only = TRUE)))
message("OK: ", paste(needed, collapse = ", "))
