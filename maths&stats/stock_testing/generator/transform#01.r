library(ggplot2)
library(dplyr)
library(zoo)

# Read data
df <- read.csv("C:/Users/Admin/Documents/r learning/generated_data.csv", stringsAsFactors = FALSE)
df$Date <- as.POSIXct(df$Date)
df <- df[order(df$Date), ]

# Use consecutive positions so overnight/weekend gaps are compressed
df$plot_x <- seq_len(nrow(df))

# One x-axis label per date, based on timestamps read from the CSV
daily_ticks <- df %>%
  group_by(day = as.Date(Date)) %>%
  summarise(x = mean(range(plot_x)), .groups = "drop")

# Parameters
fast_n <- 5
slow_n <- 12
ci_level <- 0.95
z <- qnorm(0.5 + ci_level / 2)

# Moving averages
df$SMA_fast <- zoo::rollmean(df$Close, k = fast_n, fill = NA, align = "right")
df$SMA_slow <- zoo::rollmean(df$Close, k = slow_n, fill = NA, align = "right")

# Rolling SD for CI (aligned with slow MA)
df$roll_sd <- zoo::rollapply(df$Close, width = slow_n, FUN = sd, fill = NA, align = "right")
df$CI_upper <- df$SMA_slow + z * df$roll_sd
df$CI_lower <- df$SMA_slow - z * df$roll_sd

# Candle direction
df$direction <- ifelse(df$Close >= df$Open, "up", "down")

# Plot
print(ggplot(df, aes(x = plot_x)) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper), fill = "#313182", alpha = 0.15) +
  geom_segment(aes(xend = plot_x, y = Low, yend = High), color = "black", linewidth = 0.4) +
  geom_rect(aes(xmin = plot_x - 0.35, xmax = plot_x + 0.35,
                ymin = pmin(Open, Close), ymax = pmax(Open, Close),
                fill = direction), color = "black", linewidth = 0.2) +
  geom_line(aes(y = SMA_fast), color = "blue", linewidth = 0.8) +
  geom_line(aes(y = SMA_slow), color = "#ffd900", linewidth = 0.8) +
  scale_x_continuous(
    breaks = daily_ticks$x,
    labels = format(daily_ticks$day, "%A\n%d %b"),
    expand = c(0, 0)
  ) +
  scale_fill_manual(values = c("up" = "#26a69a", "down" = "#ef5350")) +
  labs(title = "Candlestick Chart with 95% CI and Moving Averages",
       subtitle = paste("Fast SMA (", fast_n, ") = Blue | Slow SMA (", slow_n,
                        ") = Yellow | 95% CI Band", sep = ""),
       x = "Date (from CSV)", y = "Price") +
  theme_minimal() +
  theme(legend.position = "none")
)
