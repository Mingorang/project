data <- read.csv("generated_data.csv")

cutoff <- 1
mode <- "inside"

filtered_data <- data |>
  mutate(
    open_close = 100 * ((Close - Open) / Open)
  ) |>
  filter(
    if (mode == "inside") {
      -cutoff < open_close & open_close < cutoff
    } else {
      open_close > cutoff | open_close < -cutoff
    }
  )

print(filtered_data)
# look at the cutoff variable, could I add a loop that iterates until n 
#(n for say 25) filtered results appear, so if cutoff was 0, all 45-50 would 
#appear but if cutoff was 6 there 0 would appear. use this approach to do something

