library(readr)
library(dplyr)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1],show_col_types=FALSE)
# CSV columns: regime, loss
print(x %>% group_by(regime) %>% summarise(n=n(),mean_loss=mean(loss,na.rm=TRUE),.groups="drop"))
