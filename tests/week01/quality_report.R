library(readr)
library(dplyr)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1], show_col_types=FALSE)
print(x %>% summarise(rows=n(), missing_close=sum(is.na(Close)), duplicates=sum(duplicated(Date))))
