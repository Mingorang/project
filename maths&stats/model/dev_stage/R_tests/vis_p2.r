library(ggplot2)
data <- read.csv("generated_data.csv")
facet_wrap(data)
ggplot(data) +
  stat_smooth(aes(x = Date, y = Open, group = 1), linetype = "solid", level = 0.7) +
  scale_x_discrete(breaks = data$Date[seq(1, nrow(data), length.out = 5)]) +
  theme(axis.text.x = element_text(angle = 41, hjust = 0.9, size=6)) +
  labs(
    x = "Days since model start",
    y = "Stock price"
  )
#geom_point()+


theme()