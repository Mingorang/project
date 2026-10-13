library(readr)
library(dplyr)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1],show_col_types=FALSE)
# CSV columns: model, mae, similarity
print(x %>% group_by(model) %>% summarise(mean_mae=mean(mae),mean_similarity=mean(similarity),.groups="drop"))
