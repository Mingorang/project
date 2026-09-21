library(dplyr)
data <- read.csv("generated_data.csv")
data|> arrange(desc(Date))
data %>%
  filter( Open < 102) # filters
data %>%
  select(Date, Close) 
data %>%
  mutate(  # Make new variables and add to a csv ect
    percent_diff = 100*(Close-Open)/Open
  )
data %>%
  arrange(desc(Date)) # Adds order, desc is descending
data %>%
  summarise(
    Mean_Open = mean(Open)
  )# Summary, not for adding variables
data %>%
  group_by(Date) %>%
  summarise(
    Mean_Close = mean(Close)
  )
