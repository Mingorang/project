library(tidyverse)
library(lubridate)
library(dplyr)
stocks <- read_csv("generated_data.csv")
print(stocks)
readr::write_csv(stocks, "output_data.csv")
stocks %>% 
  mutate(wday = wday(Date))
stocks %>% 
  mutate(wday = wday(Date, label = TRUE, abbr = FALSE))
