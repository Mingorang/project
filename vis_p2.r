library(ggplot2)
library(dplyr)
library(lubridate)
data <- read.csv("generated_data.csv")
data <- data |> 
  mutate(
    Date = as.Date(Date),
    wday = wday(Date, label = FALSE, week_start = 1) 
  )
print(ggplot(data) +
  stat_smooth(aes(x = wday, y = Open, group = 1), linetype = "solid", level = 0.7) +
  # Force the X-axis to show sequence 1 to 7 corresponding to Mon-Sun
  scale_x_continuous(
    breaks = 1:7,
    labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
  ) +
  theme(axis.text.x = element_text(angle = 41, hjust = 0.9, size = 8)) +
  labs(
    x = "Day of the week",
    y = "Stock price"
  ))