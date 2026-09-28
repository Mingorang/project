library(ggplot2)
data <- read.csv("generated_data.csv")

ggplot(data, aes(Date, Close)) +
  geom_line()

geom_line()
geom_point()
geom_histogram()
geom_density()
geom_boxplot()
geom_smooth()