data <- read.csv("generated_data.csv")
ahh <- length(col(data))/length(data)
library(ggplot2)
ggplot(data, aes(y = Open, x = 0:(ahh-1))) +
  geom_line()

