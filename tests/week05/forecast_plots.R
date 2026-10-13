library(readr)
library(ggplot2)
x <- read_csv(commandArgs(trailingOnly=TRUE)[1],show_col_types=FALSE)
# CSV columns: timestamp, actual, predicted
print(ggplot(x,aes(x=timestamp))+geom_line(aes(y=actual))+geom_line(aes(y=predicted),linetype=2)+labs(y="Log price",title="Observed vs simulated forecast"))
