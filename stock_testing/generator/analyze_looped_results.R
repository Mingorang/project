#!/usr/bin/env Rscript

# Batch-analyze data_N.csv and image_N.png pairs from looped_results.
# The best historical trades are hindsight measurements, not forecasts.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else "analyze_looped_results.R"
model_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
input_dir <- file.path(model_dir, "looped_results")
output_dir <- file.path(model_dir, "looped_results_analysis_300")
input_source <- "working-tree files"

if (!dir.exists(input_dir)) {
  repo_root <- tryCatch(
    system2("git", c("-C", shQuote(model_dir), "rev-parse", "--show-toplevel"), stdout = TRUE, stderr = FALSE),
    error = function(error) character()
  )
  if (!length(repo_root)) stop("Input folder is missing and Git repository root could not be determined.")

  archive_path <- tempfile(fileext = ".tar")
  archive_root <- tempfile("looped-results-")
  dir.create(archive_root)
  archive_status <- system2(
    "git",
    c("-C", shQuote(repo_root[1L]), "archive", "HEAD", "maths&stats/model/looped_results"),
    stdout = archive_path,
    stderr = FALSE
  )
  if (!identical(as.integer(archive_status), 0L)) {
    stop("Input folder is missing and committed looped_results files could not be read from Git HEAD.")
  }
  utils::untar(archive_path, exdir = archive_root)
  input_dir <- file.path(archive_root, "maths&stats", "model", "looped_results")
  if (!dir.exists(input_dir)) stop("No looped_results folder found in the Git HEAD archive.")
  input_source <- "Git HEAD archive (working-tree files were left untouched)"
}

required_packages <- c("ggplot2", "png")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install required R packages first: install.packages(c(",
       paste(sprintf("\"%s\"", missing_packages), collapse = ", "), "))")
}
if (!dir.exists(input_dir)) stop("Input folder not found: ", input_dir)
if (!dir.exists(output_dir)) {
  stop("Analysis output folder not found (not creating it): ", output_dir)
}

fast_n <- 5L
slow_n <- 12L
confidence_level <- 0.95
target_multiple <- 2

csv_paths <- list.files(input_dir, pattern = "^data_[0-9]+\\.csv$", full.names = TRUE)
png_paths <- list.files(input_dir, pattern = "^image_[0-9]+\\.png$", full.names = TRUE)
csv_ids <- as.integer(sub("^data_([0-9]+)\\.csv$", "\\1", basename(csv_paths)))
png_ids <- as.integer(sub("^image_([0-9]+)\\.png$", "\\1", basename(png_paths)))
run_ids <- sort(intersect(csv_ids, png_ids))

if (!length(run_ids)) stop("No matching data_N.csv and image_N.png pairs found in ", input_dir)
if (length(setdiff(csv_ids, png_ids))) {
  warning("CSV files without PNG pairs: ", paste(setdiff(csv_ids, png_ids), collapse = ", "))
}
if (length(setdiff(png_ids, csv_ids))) {
  warning("PNG files without CSV pairs: ", paste(setdiff(png_ids, csv_ids), collapse = ", "))
}

csv_by_id <- setNames(csv_paths, csv_ids)
png_by_id <- setNames(png_paths, png_ids)
rolling_mean <- function(values, width) {
  result <- rep(NA_real_, length(values))
  if (length(values) >= width) {
    for (index in seq.int(width, length(values))) {
      result[index] <- mean(values[seq.int(index - width + 1L, index)])
    }
  }
  result
}

rolling_sd <- function(values, width) {
  result <- rep(NA_real_, length(values))
  if (length(values) >= width) {
    for (index in seq.int(width, length(values))) {
      result[index] <- stats::sd(values[seq.int(index - width + 1L, index)])
    }
  }
  result
}

wilder_smooth <- function(values, period) {
  result <- rep(NA_real_, length(values))
  valid <- which(is.finite(values))
  if (length(valid) < period) return(result)
  seed_rows <- valid[seq_len(period)]
  first_row <- seed_rows[period]
  result[first_row] <- mean(values[seed_rows])
  if (first_row < length(values)) {
    for (index in seq.int(first_row + 1L, length(values))) {
      if (is.finite(values[index])) {
        result[index] <- (result[index - 1L] * (period - 1L) + values[index]) / period
      }
    }
  }
  result
}

ema_smooth <- function(values, period) {
  result <- rep(NA_real_, length(values))
  valid <- which(is.finite(values))
  if (length(valid) < period) return(result)
  seed_rows <- valid[seq_len(period)]
  first_row <- seed_rows[period]
  result[first_row] <- mean(values[seed_rows])
  alpha <- 2 / (period + 1)
  if (first_row < length(values)) {
    for (index in seq.int(first_row + 1L, length(values))) {
      if (is.finite(values[index])) {
        result[index] <- alpha * values[index] + (1 - alpha) * result[index - 1L]
      }
    }
  }
  result
}

save_three_panel_plot <- function(price_plot, rsi_plot, macd_plot, output_path) {
  grDevices::png(output_path, width = 1800, height = 1350, res = 150, bg = "white")
  on.exit(grDevices::dev.off(), add = TRUE)
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(
    layout = grid::grid.layout(3L, 1L, heights = grid::unit(c(0.62, 0.18, 0.20), "null"))
  ))
  plots <- list(price_plot, rsi_plot, macd_plot)
  for (row in seq_along(plots)) {
    grid::pushViewport(grid::viewport(layout.pos.row = row, layout.pos.col = 1L))
    grid::grid.draw(ggplot2::ggplotGrob(plots[[row]]))
    grid::popViewport()
  }
  grid::popViewport()
}

detect_crossovers <- function(data) {
  difference <- data$fast_ma - data$slow_ma
  signals <- data.frame(signal_index = integer(), side = character())
  if (nrow(data) < 2L) return(signals)

  for (index in seq.int(2L, nrow(data))) {
    previous <- difference[index - 1L]
    current <- difference[index]
    if (!is.finite(previous) || !is.finite(current)) next
    if (previous <= 0 && current > 0) {
      signals <- rbind(signals, data.frame(signal_index = index, side = "Long"))
    } else if (previous >= 0 && current < 0) {
      signals <- rbind(signals, data.frame(signal_index = index, side = "Short"))
    }
  }
  signals
}

simulate_crossover_trade <- function(data, signal_index, side, run_id) {
  entry_index <- signal_index + 1L
  if (entry_index > nrow(data)) return(NULL)

  entry_price <- data$Open[entry_index]
  stop_price <- if (side == "Long") data$Low[signal_index] else data$High[signal_index]
  risk_per_unit <- if (side == "Long") entry_price - stop_price else stop_price - entry_price
  if (!is.finite(risk_per_unit) || risk_per_unit <= 0) return(NULL)

  target_price <- if (side == "Long") {
    entry_price + target_multiple * risk_per_unit
  } else {
    entry_price - target_multiple * risk_per_unit
  }
  exit_index <- nrow(data)
  exit_price <- data$Close[exit_index]
  exit_reason <- "time_exit"

  for (index in seq.int(entry_index, nrow(data))) {
    open_price <- data$Open[index]
    if (side == "Long" && open_price <= stop_price) {
      exit_index <- index
      exit_price <- open_price
      exit_reason <- "stop_gap"
      break
    }
    if (side == "Short" && open_price >= stop_price) {
      exit_index <- index
      exit_price <- open_price
      exit_reason <- "stop_gap"
      break
    }
    if (side == "Long" && open_price >= target_price) {
      exit_index <- index
      exit_price <- open_price
      exit_reason <- "target_gap"
      break
    }
    if (side == "Short" && open_price <= target_price) {
      exit_index <- index
      exit_price <- open_price
      exit_reason <- "target_gap"
      break
    }

    stop_hit <- if (side == "Long") data$Low[index] <= stop_price else data$High[index] >= stop_price
    target_hit <- if (side == "Long") data$High[index] >= target_price else data$Low[index] <= target_price
    if (stop_hit) {
      exit_index <- index
      exit_price <- stop_price
      exit_reason <- "stop"
      break
    }
    if (target_hit) {
      exit_index <- index
      exit_price <- target_price
      exit_reason <- "target"
      break
    }
  }

  pnl <- if (side == "Long") exit_price - entry_price else entry_price - exit_price
  data.frame(
    run = run_id,
    side = side,
    signal_index = signal_index,
    signal_time = data$Date[signal_index],
    entry_index = entry_index,
    entry_time = data$Date[entry_index],
    entry_price = entry_price,
    stop_price = stop_price,
    target_price = target_price,
    risk_per_unit = risk_per_unit,
    exit_index = exit_index,
    exit_time = data$Date[exit_index],
    exit_price = exit_price,
    pnl_per_unit = pnl,
    r_multiple = pnl / risk_per_unit,
    exit_reason = exit_reason,
    stringsAsFactors = FALSE
  )
}

best_trade <- function(trades, side) {
  candidates <- trades[trades$side == side, , drop = FALSE]
  if (!nrow(candidates)) return(candidates)
  candidates[order(-candidates$r_multiple, candidates$entry_index)[1L], , drop = FALSE]
}

trade_value <- function(trade, column) {
  if (!nrow(trade)) return(NA_real_)
  as.numeric(trade[[column]][1L])
}

trade_time <- function(trade, column) {
  if (!nrow(trade)) return(NA_character_)
  format(trade[[column]][1L], "%Y-%m-%d %H:%M:%S", tz = "UTC")
}

make_trade_markers <- function(long_trade, short_trade) {
  markers <- list()
  for (trade in list(long_trade, short_trade)) {
    if (!nrow(trade)) next
    prefix <- if (trade$side[1L] == "Long") "L" else "S"
    markers[[length(markers) + 1L]] <- data.frame(
      bar = trade$entry_index[1L], price = trade$entry_price[1L], side = trade$side[1L],
      label = paste0(prefix, " in ", sprintf("%+.2fR", trade$r_multiple[1L])),
      vjust = if (prefix == "L") -0.7 else 1.5
    )
    markers[[length(markers) + 1L]] <- data.frame(
      bar = trade$exit_index[1L], price = trade$exit_price[1L], side = trade$side[1L],
      label = paste0(prefix, " out"),
      vjust = if (prefix == "L") 1.5 else -0.7
    )
  }
  if (!length(markers)) {
    return(data.frame(bar = numeric(), price = numeric(), side = character(), label = character(), vjust = numeric()))
  }
  do.call(rbind, markers)
}

summary_rows <- vector("list", length(run_ids))
processed_bar_rows <- vector("list", length(run_ids))
all_trade_rows <- list()

for (run_position in seq_along(run_ids)) {
  run_id <- run_ids[run_position]
  csv_path <- unname(csv_by_id[as.character(run_id)])
  png_path <- unname(png_by_id[as.character(run_id)])
  data <- utils::read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
  required_columns <- c("Date", "Open", "High", "Low", "Close")
  if (!all(required_columns %in% names(data))) {
    stop("Missing required OHLC columns in ", basename(csv_path))
  }
  data$Date <- as.POSIXct(data$Date, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  if (anyNA(data$Date)) stop("Could not parse all timestamps in ", basename(csv_path))
  for (column in c("Open", "High", "Low", "Close")) {
    data[[column]] <- suppressWarnings(as.numeric(data[[column]]))
    if (any(!is.finite(data[[column]]))) stop("Non-numeric or missing ", column, " values in ", basename(csv_path))
  }
  data <- data[order(data$Date), , drop = FALSE]
  rownames(data) <- NULL
  if (nrow(data) < slow_n) stop("Need at least ", slow_n, " rows in ", basename(csv_path))
  if (anyDuplicated(data$Date)) stop("Duplicate timestamps in ", basename(csv_path))
  if (any(data$High < pmax(data$Open, data$Close)) || any(data$Low > pmin(data$Open, data$Close))) {
    stop("Invalid OHLC candle ordering in ", basename(csv_path))
  }

  data$bar <- seq_len(nrow(data))
  data$fast_ma <- rolling_mean(data$Close, fast_n)
  data$slow_ma <- rolling_mean(data$Close, slow_n)
  rolling_deviation <- rolling_sd(data$Close, slow_n)
  critical_value <- stats::qt(1 - (1 - confidence_level) / 2, df = slow_n - 1L)
  ci_margin <- critical_value * rolling_deviation / sqrt(slow_n)
  data$ci_lower <- data$slow_ma - ci_margin
  data$ci_upper <- data$slow_ma + ci_margin
  data$bb_mid_20 <- rolling_mean(data$Close, 20L)
  bb_sd <- rolling_sd(data$Close, 20L)
  data$bb_upper_20_2sd <- data$bb_mid_20 + 2 * bb_sd
  data$bb_lower_20_2sd <- data$bb_mid_20 - 2 * bb_sd
  bb_width <- data$bb_upper_20_2sd - data$bb_lower_20_2sd
  data$bb_percent_b <- (data$Close - data$bb_lower_20_2sd) / ifelse(bb_width == 0, NA_real_, bb_width)
  data$bb_width_pct <- 100 * bb_width / ifelse(data$bb_mid_20 == 0, NA_real_, abs(data$bb_mid_20))
  price_change <- c(NA_real_, diff(data$Close))
  average_gain <- wilder_smooth(pmax(price_change, 0), 14L)
  average_loss <- wilder_smooth(pmax(-price_change, 0), 14L)
  relative_strength <- average_gain / ifelse(average_loss == 0, NA_real_, average_loss)
  data$rsi_14 <- 100 - 100 / (1 + relative_strength)
  data$rsi_14[is.finite(average_gain) & average_loss == 0 & average_gain > 0] <- 100
  data$rsi_14[is.finite(average_gain) & average_loss == 0 & average_gain == 0] <- 50
  data$macd_12_26 <- ema_smooth(data$Close, 12L) - ema_smooth(data$Close, 26L)
  data$macd_signal_9 <- ema_smooth(data$macd_12_26, 9L)
  data$macd_histogram <- data$macd_12_26 - data$macd_signal_9
  previous_close <- c(NA_real_, head(data$Close, -1L))
  data$true_range <- pmax(
    data$High - data$Low,
    abs(data$High - previous_close),
    abs(data$Low - previous_close),
    na.rm = TRUE
  )
  data$true_range[1L] <- data$High[1L] - data$Low[1L]
  data$atr_14 <- wilder_smooth(data$true_range, 14L)
  data$bar_change <- price_change
  data$bar_return_pct <- ifelse(previous_close > 0, 100 * (data$Close / previous_close - 1), NA_real_)
  data$direction <- ifelse(data$Close >= data$Open, "Up", "Down")
  data$run <- run_id
  processed_bar_rows[[run_position]] <- data

  signals <- detect_crossovers(data)
  run_trades <- list()
  if (nrow(signals)) {
    for (signal_position in seq_len(nrow(signals))) {
      trade <- simulate_crossover_trade(
        data,
        signals$signal_index[signal_position],
        signals$side[signal_position],
        run_id
      )
      if (!is.null(trade)) run_trades[[length(run_trades) + 1L]] <- trade
    }
  }
  trades <- if (length(run_trades)) do.call(rbind, run_trades) else data.frame(
    run = integer(), side = character(), signal_index = integer(), signal_time = as.POSIXct(character()),
    entry_index = integer(), entry_time = as.POSIXct(character()), entry_price = numeric(),
    stop_price = numeric(), target_price = numeric(), risk_per_unit = numeric(),
    exit_index = integer(), exit_time = as.POSIXct(character()), exit_price = numeric(),
    pnl_per_unit = numeric(), r_multiple = numeric(), exit_reason = character()
  )
  if (nrow(trades)) all_trade_rows[[length(all_trade_rows) + 1L]] <- trades

  long_best <- best_trade(trades, "Long")
  short_best <- best_trade(trades, "Short")
  valid_returns <- data$bar_return_pct[is.finite(data$bar_return_pct)] / 100
  negative_returns <- valid_returns[valid_returns < 0]
  return_sd <- if (length(valid_returns) > 1L) stats::sd(valid_returns) else NA_real_
  downside_deviation <- if (length(negative_returns)) sqrt(mean(negative_returns ^ 2)) else NA_real_
  sharpe_per_bar <- if (is.finite(return_sd) && return_sd > 0) mean(valid_returns) / return_sd else NA_real_
  sortino_per_bar <- if (is.finite(downside_deviation) && downside_deviation > 0) mean(valid_returns) / downside_deviation else NA_real_
  close_drawdown <- cummax(data$Close) - data$Close
  close_drawdown_pct <- if (all(data$Close > 0)) min(data$Close / cummax(data$Close) - 1) * 100 else NA_real_
  candle_range <- data$High - data$Low
  candle_body <- abs(data$Close - data$Open)
  bar_gap <- data$Open - previous_close
  total_return_pct <- if (data$Open[1L] > 0) (data$Close[nrow(data)] / data$Open[1L] - 1) * 100 else NA_real_
  day_values <- as.Date(data$Date, tz = "UTC")
  first_day_rows <- which(!duplicated(day_values))
  markers <- make_trade_markers(long_best, short_best)

  price_plot <- ggplot2::ggplot(data, ggplot2::aes(x = bar)) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = ci_lower, ymax = ci_upper),
      fill = "#75aadb", alpha = 0.24, na.rm = TRUE
    ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = bb_lower_20_2sd, ymax = bb_upper_20_2sd),
      fill = "#aeb7c2", alpha = 0.13, na.rm = TRUE
    ) +
    ggplot2::geom_hline(yintercept = 0, color = "grey55", linetype = "dotted", linewidth = 0.35) +
    ggplot2::geom_segment(ggplot2::aes(xend = bar, y = Low, yend = High), color = "grey25", linewidth = 0.35) +
    ggplot2::geom_rect(
      ggplot2::aes(
        xmin = bar - 0.32, xmax = bar + 0.32,
        ymin = pmin(Open, Close), ymax = pmax(Open, Close), fill = direction
      ),
      color = "grey20", linewidth = 0.2
    ) +
    ggplot2::geom_line(ggplot2::aes(y = fast_ma), color = "#1769aa", linewidth = 0.7, na.rm = TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = slow_ma), color = "#d28b00", linewidth = 0.8, na.rm = TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = bb_mid_20), color = "#606b78", linewidth = 0.6, linetype = "dashed", na.rm = TRUE) +
    ggplot2::scale_fill_manual(values = c(Up = "#26a69a", Down = "#ef5350")) +
    ggplot2::scale_x_continuous(
      breaks = data$bar[first_day_rows],
      labels = format(day_values[first_day_rows], "%a %d %b"),
      expand = ggplot2::expansion(mult = c(0.01, 0.02))
    ) +
    ggplot2::labs(
      title = sprintf("Run %d: OHLC, SMA and Bollinger Bands", run_id),
      subtitle = "Blue: 95% rolling t-interval for 12-close mean | Blue line: SMA 5 | Gold line: SMA 12 | Grey band: Bollinger 20 +/- 2 SD",
      x = NULL, y = "Price", fill = "Candle"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 25, hjust = 1),
      legend.position = "bottom"
    )

  if (nrow(markers)) {
    price_plot <- price_plot +
      ggplot2::geom_point(
        data = markers,
        ggplot2::aes(x = bar, y = price, color = side),
        inherit.aes = FALSE, size = 2.1
      ) +
      ggplot2::geom_text(
        data = markers,
        ggplot2::aes(x = bar, y = price, label = label, color = side, vjust = vjust),
        inherit.aes = FALSE, size = 2.7, show.legend = FALSE
      ) +
      ggplot2::scale_color_manual(values = c(Long = "#087f5b", Short = "#c92a2a"))
  }

  long_label <- if (nrow(long_best)) sprintf("Long best: %.2fR", long_best$r_multiple[1L]) else "Long best: no valid crossover"
  short_label <- if (nrow(short_best)) sprintf("Short best: %.2fR", short_best$r_multiple[1L]) else "Short best: no valid crossover"
  price_plot <- price_plot + ggplot2::labs(
  )
  rsi_plot <- ggplot2::ggplot(data, ggplot2::aes(x = bar, y = rsi_14)) +
    ggplot2::geom_line(color = "#7651a8", linewidth = 0.7, na.rm = TRUE) +
    ggplot2::geom_hline(yintercept = c(30, 70), linetype = "dashed", color = "#777777", linewidth = 0.4) +
    ggplot2::scale_x_continuous(breaks = data$bar[first_day_rows], labels = format(day_values[first_day_rows], "%a %d %b")) +
    ggplot2::coord_cartesian(ylim = c(0, 100)) +
    ggplot2::labs(title = "RSI (14)", subtitle = "Study [4]: Relative Strength Index", x = NULL, y = "RSI") +
    ggplot2::theme_minimal(base_size = 8) + ggplot2::theme(axis.text.x = ggplot2::element_blank(), axis.ticks.x = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank())
  macd_plot <- ggplot2::ggplot(data, ggplot2::aes(x = bar)) +
    ggplot2::geom_col(ggplot2::aes(y = macd_histogram), fill = "#aeb7c2", width = 0.7, na.rm = TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = macd_12_26), color = "#1769aa", linewidth = 0.6, na.rm = TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = macd_signal_9), color = "#d28b00", linewidth = 0.6, na.rm = TRUE) +
    ggplot2::geom_hline(yintercept = 0, color = "#777777", linewidth = 0.4) +
    ggplot2::scale_x_continuous(breaks = data$bar[first_day_rows], labels = format(day_values[first_day_rows], "%a %d %b")) +
    ggplot2::labs(title = "MACD (12, 26, 9)", subtitle = "Study [5]: MACD line, signal line and histogram", x = "Trading session (overnight gaps compressed)", y = "MACD") +
    ggplot2::theme_minimal(base_size = 8) + ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  save_three_panel_plot(
    price_plot, rsi_plot, macd_plot,
    file.path(output_dir, sprintf("image_%d_analysis.png", run_id))
  )

  summary_rows[[run_position]] <- data.frame(
    run = run_id,
    source_csv = basename(csv_path),
    source_png = basename(png_path),
    input_source = input_source,
    bars = nrow(data),
    fast_window = fast_n,
    slow_window = slow_n,
    confidence_level = confidence_level,
    first_open = data$Open[1L],
    last_close = data$Close[nrow(data)],
    net_change_points = data$Close[nrow(data)] - data$Open[1L],
    total_return_pct = total_return_pct,
    min_low = min(data$Low),
    max_high = max(data$High),
    high_low_range_points = max(data$High) - min(data$Low),
    high_low_range_pct_of_low = if (min(data$Low) > 0) (max(data$High) - min(data$Low)) / min(data$Low) * 100 else NA_real_,
    mean_close = mean(data$Close),
    median_close = stats::median(data$Close),
    mean_bar_change_points = mean(data$bar_change, na.rm = TRUE),
    median_bar_change_points = stats::median(data$bar_change, na.rm = TRUE),
    sd_bar_change_points = stats::sd(data$bar_change, na.rm = TRUE),
    mean_valid_bar_return_pct = mean(data$bar_return_pct, na.rm = TRUE),
    median_valid_bar_return_pct = stats::median(data$bar_return_pct, na.rm = TRUE),
    realized_vol_per_bar_pct = stats::sd(data$bar_return_pct, na.rm = TRUE),
    valid_return_observations = sum(is.finite(data$bar_return_pct)),
    sharpe_per_bar_no_rf = sharpe_per_bar,
    sortino_per_bar_no_rf = sortino_per_bar,
    max_drawdown_points = max(close_drawdown),
    max_drawdown_pct_positive_prices_only = close_drawdown_pct,
    bullish_candles = sum(data$direction == "Up"),
    bearish_candles = sum(data$direction == "Down"),
    bullish_candle_pct = mean(data$direction == "Up") * 100,
    bearish_candle_pct = mean(data$direction == "Down") * 100,
    avg_candle_body_points = mean(candle_body),
    avg_candle_range_points = mean(candle_range),
    median_candle_range_points = stats::median(candle_range),
    avg_true_range_points = mean(data$true_range),
    atr14_last = tail(data$atr_14, 1L),
    rsi14_last = tail(data$rsi_14, 1L),
    macd12_26_last = tail(data$macd_12_26, 1L),
    macd_signal9_last = tail(data$macd_signal_9, 1L),
    macd_histogram_last = tail(data$macd_histogram, 1L),
    bollinger20_upper_last = tail(data$bb_upper_20_2sd, 1L),
    bollinger20_lower_last = tail(data$bb_lower_20_2sd, 1L),
    bollinger_percent_b_last = tail(data$bb_percent_b, 1L),
    bollinger_width_pct_last = tail(data$bb_width_pct, 1L),
    sma_bull_crossovers_5_12 = sum(signals$side == "Long"),
    sma_bear_crossovers_5_12 = sum(signals$side == "Short"),
    gap_bars = sum(abs(bar_gap) > 1e-9, na.rm = TRUE),
    avg_abs_gap_points = mean(abs(bar_gap), na.rm = TRUE),
    winning_crossover_trades = sum(trades$r_multiple > 0),
    valid_crossover_trades = nrow(trades),
    long_crossover_trades = sum(trades$side == "Long"),
    short_crossover_trades = sum(trades$side == "Short"),
    best_long_entry = trade_time(long_best, "entry_time"),
    best_long_exit = trade_time(long_best, "exit_time"),
    best_long_entry_price = trade_value(long_best, "entry_price"),
    best_long_stop = trade_value(long_best, "stop_price"),
    best_long_target = trade_value(long_best, "target_price"),
    best_long_r_multiple = trade_value(long_best, "r_multiple"),
    best_long_pnl_per_unit = trade_value(long_best, "pnl_per_unit"),
    best_short_entry = trade_time(short_best, "entry_time"),
    best_short_exit = trade_time(short_best, "exit_time"),
    best_short_entry_price = trade_value(short_best, "entry_price"),
    best_short_stop = trade_value(short_best, "stop_price"),
    best_short_target = trade_value(short_best, "target_price"),
    best_short_r_multiple = trade_value(short_best, "r_multiple"),
    best_short_pnl_per_unit = trade_value(short_best, "pnl_per_unit"),
    stringsAsFactors = FALSE
  )
}

summary_table <- do.call(rbind, summary_rows)
utils::write.csv(summary_table, file.path(output_dir, "analysis_summary.csv"), row.names = FALSE, na = "")
utils::write.csv(do.call(rbind, processed_bar_rows), file.path(output_dir, "processed_bars.csv"), row.names = FALSE, na = "")
trade_table <- if (length(all_trade_rows)) do.call(rbind, all_trade_rows) else data.frame(
  run = integer(), side = character(), signal_index = integer(), signal_time = character(),
  entry_index = integer(), entry_time = character(), entry_price = numeric(),
  stop_price = numeric(), target_price = numeric(), risk_per_unit = numeric(),
  exit_index = integer(), exit_time = character(), exit_price = numeric(),
  pnl_per_unit = numeric(), r_multiple = numeric(), exit_reason = character()
)
utils::write.csv(trade_table, file.path(output_dir, "crossover_trades.csv"), row.names = FALSE, na = "")

all_bar_data <- do.call(rbind, processed_bar_rows)
highest_row <- all_bar_data[which.max(all_bar_data$High), , drop = FALSE]
lowest_row <- all_bar_data[which.min(all_bar_data$Low), , drop = FALSE]
best_long_overall <- trade_table[trade_table$side == "Long", , drop = FALSE]
best_short_overall <- trade_table[trade_table$side == "Short", , drop = FALSE]
if (nrow(best_long_overall)) best_long_overall <- best_long_overall[which.max(best_long_overall$r_multiple), , drop = FALSE]
if (nrow(best_short_overall)) best_short_overall <- best_short_overall[which.max(best_short_overall$r_multiple), , drop = FALSE]
metric_row <- function(measure, value, context = "") {
  data.frame(measure = measure, value = as.character(value), context = context, stringsAsFactors = FALSE)
}
overall_metrics <- do.call(rbind, list(
  metric_row("run_pairs", length(run_ids), "count"),
  metric_row("ohlc_bars", nrow(all_bar_data), "count"),
  metric_row("unpaired_csv", length(setdiff(csv_ids, png_ids)), "count"),
  metric_row("unpaired_png", length(setdiff(png_ids, csv_ids)), "count"),
  metric_row("invalid_ohlc_rows", sum(all_bar_data$High < pmax(all_bar_data$Open, all_bar_data$Close) | all_bar_data$Low > pmin(all_bar_data$Open, all_bar_data$Close)), "count"),
  metric_row("highest_high", highest_row$High, paste("run", highest_row$run, "at", highest_row$Date)),
  metric_row("lowest_low", lowest_row$Low, paste("run", lowest_row$run, "at", lowest_row$Date)),
  metric_row("average_run_return_pct", mean(summary_table$total_return_pct, na.rm = TRUE), "percent"),
  metric_row("median_run_return_pct", stats::median(summary_table$total_return_pct, na.rm = TRUE), "percent"),
  metric_row("positive_return_runs", sum(summary_table$total_return_pct > 0, na.rm = TRUE), "count"),
  metric_row("negative_return_runs", sum(summary_table$total_return_pct < 0, na.rm = TRUE), "count"),
  metric_row("median_realized_volatility_per_bar_pct", stats::median(summary_table$realized_vol_per_bar_pct, na.rm = TRUE), "percent, unannualized"),
  metric_row("median_high_low_range_points", stats::median(summary_table$high_low_range_points, na.rm = TRUE), "price points"),
  metric_row("mean_bullish_candle_share_pct", mean(summary_table$bullish_candle_pct, na.rm = TRUE), "percent"),
  metric_row("mean_bearish_candle_share_pct", mean(summary_table$bearish_candle_pct, na.rm = TRUE), "percent"),
  metric_row("total_bullish_sma_crossovers", sum(summary_table$sma_bull_crossovers_5_12), "count"),
  metric_row("total_bearish_sma_crossovers", sum(summary_table$sma_bear_crossovers_5_12), "count"),
  metric_row("candidate_crossover_trades", nrow(trade_table), "count"),
  metric_row("best_long_r_multiple", if (nrow(best_long_overall)) best_long_overall$r_multiple else NA, if (nrow(best_long_overall)) paste("run", best_long_overall$run) else "none"),
  metric_row("best_short_r_multiple", if (nrow(best_short_overall)) best_short_overall$r_multiple else NA, if (nrow(best_short_overall)) paste("run", best_short_overall$run) else "none"),
  metric_row("mean_atr14_last", mean(summary_table$atr14_last, na.rm = TRUE), "price points"),
  metric_row("mean_rsi14_last", mean(summary_table$rsi14_last, na.rm = TRUE), "0-100 scale"),
  metric_row("mean_macd_histogram_last", mean(summary_table$macd_histogram_last, na.rm = TRUE), "price points"),
  metric_row("mean_bollinger_width_last_pct", mean(summary_table$bollinger_width_pct_last, na.rm = TRUE), "percent"),
  metric_row("volume_indicators", "not available", "input CSVs contain no Volume column")
))
utils::write.csv(overall_metrics, file.path(output_dir, "overall_metrics.csv"), row.names = FALSE, na = "")

make_contact_sheet <- function(paths, ids, output_path) {
  columns <- 8L
  tile_width <- 300L
  tile_height <- 180L
  rows <- ceiling(length(paths) / columns)
  grDevices::png(
    filename = output_path,
    width = columns * tile_width,
    height = rows * tile_height,
    res = 120,
    bg = "white"
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(rows, columns)))

  for (index in seq_along(paths)) {
    row <- ceiling(index / columns)
    column <- ((index - 1L) %% columns) + 1L
    grid::pushViewport(grid::viewport(layout.pos.row = row, layout.pos.col = column))
    grid::grid.rect(gp = grid::gpar(fill = "white", col = NA))
    image <- png::readPNG(paths[index])
    grid::grid.raster(
      image,
      x = grid::unit(0.5, "npc"), y = grid::unit(0.45, "npc"),
      width = grid::unit(0.96, "npc"), height = grid::unit(0.80, "npc"),
      interpolate = TRUE
    )
    grid::grid.text(
      paste("Run", ids[index]),
      x = grid::unit(0.5, "npc"), y = grid::unit(0.94, "npc"),
      gp = grid::gpar(col = "black", fontsize = 9, fontface = "bold")
    )
    grid::popViewport()
  }
  grid::popViewport()
}

sheet_order <- order(png_ids)
make_contact_sheet(
  png_paths[sheet_order],
  png_ids[sheet_order],
  file.path(output_dir, "all_charts_contact_sheet.png")
)
analysis_png_paths <- file.path(output_dir, sprintf("image_%d_analysis.png", run_ids))
make_contact_sheet(
  analysis_png_paths,
  run_ids,
  file.path(output_dir, "all_analysis_charts_contact_sheet.png")
)

cat("Analyzed", length(run_ids), "CSV/PNG pairs.\n")
cat("Input source:", input_source, "\n")
cat("Summary:", file.path(output_dir, "analysis_summary.csv"), "\n")
cat("Per-bar indicators and confidence intervals:", file.path(output_dir, "processed_bars.csv"), "\n")
cat("Crossover trades:", file.path(output_dir, "crossover_trades.csv"), "\n")
cat("Overall metrics:", file.path(output_dir, "overall_metrics.csv"), "\n")
cat("Per-run analysis charts:", output_dir, "\n")
cat("White-background source contact sheet:", file.path(output_dir, "all_charts_contact_sheet.png"), "\n")