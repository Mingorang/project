library(readr)
library(ggplot2)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1],show_col_types=FALSE)
# CSV columns: origin, model, mae
print(ggplot(x,aes(x=origin,y=mae,colour=model))+geom_line()+labs(title="Walk-forward error"))
