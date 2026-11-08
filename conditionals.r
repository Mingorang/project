case_when(
  x < 80 ~ "right",
  x > 130 ~ "not right",
  TRUE ~ as.character(x)
)