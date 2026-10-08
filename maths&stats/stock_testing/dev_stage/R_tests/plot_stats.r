library(dplyr)
library(tidyverse)
data <- read.csv("generated_data.csv")
ggplot(data) +
  stat_smooth(aes(x = Date, y = Open, group = 1), linetype = "solid", level = 0.7) +
  scale_x_discrete(breaks = data$Date[seq(1, nrow(data), length.out = 9)]) +
  theme(axis.text.x = element_text(angle = 41, hjust = 0.97))#geom_point()
#geom_histogram()
#geom_density()
#geom_boxplot()
#geom_smooth() must be for confidence intervals
#
#ggplot(data, aes(x=Date, y=Close, group = 1)) +
#  geom_histogram()