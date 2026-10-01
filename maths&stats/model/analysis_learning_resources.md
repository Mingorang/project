# Market-Analysis Learning Links

These are public study references for the standard chart indicators and risk measures used by this project. They are not proprietary Bloomberg formulas; Bloomberg Terminal may offer comparable chart studies and additional proprietary data/tools, some requiring a subscription.

## The Chart Panels

- **[1] 5/12-period simple moving averages and crossovers:** [StockCharts ChartSchool: Moving Averages](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
- **[2] 95% rolling t-interval for a 12-close sample mean:** [NIST/SEMATECH e-Handbook: Confidence Interval Approach](https://www.itl.nist.gov/div898/handbook/prc/section2/prc221.htm)
- **[3] Bollinger Bands (20-period, 2 standard deviations):** [StockCharts ChartSchool: Bollinger Bands](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands)
- **[4] Relative Strength Index (14-period):** [StockCharts ChartSchool: RSI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
- **[5] MACD (12, 26, 9):** [StockCharts ChartSchool: MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator.md)
- **[6] Average True Range (14-period):** [StockCharts ChartSchool: ATR](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr.md)

## Performance and Risk Measures

- **[7] Sharpe ratio:** [CFA Institute: Understanding the Sharpe Ratio](https://rpc.cfainstitute.org/research/foundation/2019/understanding-the-sharpe-ratio)
- **[8] Sortino ratio:** [Investopedia: Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp)
- **[9] Maximum drawdown:** [Investopedia: Maximum Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)
- **[10] Risk/reward and R-multiples:** [Investopedia: Risk/Reward Ratio](https://www.investopedia.com/articles/trading/09/risk-reward-ratio.asp)
- **[11] Bloomberg Terminal product and charting context:** [Bloomberg Terminal](https://www.bloomberg.com/professional/products/bloomberg-terminal/)

## What the Project Calculates

The Python and R analyzers report OHLC candle counts and ranges, close-to-close changes, valid percentage returns, realized volatility, per-bar Sharpe and Sortino ratios, point and percentage drawdown where defined, bullish/bearish candle proportions, true range and ATR, SMA crossover counts, Bollinger position and width, RSI, MACD line/signal/histogram, and the best historical crossover trades under the stated stop/target rules.

The trade screen enters at the next bar's open, uses the signal candle's low/high as the stop, targets 2R, and assumes the stop is hit first if one candle touches both stop and target. The best trade is selected using later bars, so it is a hindsight comparison, not a forecast or trading recommendation. Commissions, slippage, position sizing, and taxes are not included.

A confidence interval for the rolling mean describes uncertainty about a sample mean under its assumptions; it is not a prediction interval for future prices. Price-series autocorrelation can reduce the nominal coverage of the simple rolling t-interval used here.

The CSV generator does not provide volume, so VWAP, OBV, volume profile, and volume-confirmation measures are intentionally not calculated.
