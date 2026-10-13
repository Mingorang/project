library(readr)
library(dplyr)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1], show_col_types=FALSE)
print(x %>% count(up) %>% mutate(proportion=n/sum(n)))
