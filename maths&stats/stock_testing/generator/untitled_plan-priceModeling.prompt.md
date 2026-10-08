from pathlib import Path

# Create a complete updated markdown plan. The earlier generated file could not

# be relied on after the tool failure, so this version includes the core plan

# plus the two new optional modes.

md = r"""# Price Modelling Project — PNG-Focused Research and Development Plan

## 1\. Objective

Extend the existing Python simulation -> CSV/PNG -> Python/R analysis pipeline into an empirical market-data and simulation project.

The central learning objective is to understand **everything shown in the analysis PNG**: its mathematics, interpretation, assumptions, limitations, and relationship to the underlying OHLC data.

The project should distinguish clearly between:

* observed historical/real-time market data;
* simulated data;
* descriptive statistics;
* inferential statistics;
* predictive claims.

Simulated future prices are simulations, not forecasts.

\---

Very cool, should learn R for data analysis to do something with this data
Added input to choose for real-time price 
Curb the random model for high prices, as an option for forex, relates to the manipulation comment.
need to add time module to also add real time  and price manipulation as a random event, modelled to real market manipulation
change this for forex online data for some starting price of 2 currencies, (EUR/USD) and calculate each one seperately, and plot the difference, so the plot will show the Euro dvided by the USD for a more realistic simulation


## 2\. Current Architecture

The project currently contains:

* a Python price generator;
* a loop producing numbered PNG/CSV pairs;
* a Python analysis processor;
* an R analysis processor;
* a random OHLC simulation;
* technical and statistical analysis.

Current analysis includes:

* OHLC counts and ranges;
* close-to-close changes;
* percentage returns;
* realized volatility;
* per-bar Sharpe and Sortino;
* point and percentage drawdown;
* bullish/bearish candle proportions;
* True Range and ATR;
* SMA crossover counts;
* Bollinger position and width;
* RSI;
* MACD line/signal/histogram;
* historical crossover trade analysis.

The existing learning notes correctly distinguish a confidence interval for a sample mean from a prediction interval and note that the current generator has no volume, so volume-based measures are absent.

\---

# 3\. The PNG Is the Learning Interface

Before adding major new functionality, the user should be able to look at the PNG and answer:

* What is being calculated?
* What mathematical formula is used?
* What inputs are required?
* What does a high/low value mean?
* What does an increase/decrease mean?
* What assumptions are involved?
* What can the measure tell us?
* What can it not tell us?
* Is it descriptive, inferential, or predictive?
* How can two indicators disagree without either being wrong?

Every new panel added to the PNG should have corresponding learning material.

\---

# 4\. Learning Order

Study in this order:

1. OHLC.
2. Candlesticks.
3. Price changes.
4. Percentage returns.
5. Log returns.
6. Mean and standard deviation.
7. Rolling statistics.
8. Volatility.
9. Confidence intervals.
10. Simple moving averages.
11. Moving-average crossovers.
12. Bollinger Bands.
13. True Range and ATR.
14. RSI.
15. MACD.
16. Drawdown.
17. Sharpe ratio.
18. Sortino ratio.
19. Risk/reward and R-multiples.
20. Historical trade analysis.
21. Autocorrelation.
22. Return distributions.
23. Simulation calibration.
24. Model validation.
25. Backtesting and simulation-vs-market scoring.
26. Economic-event context.
27. Web/API market-data ingestion.

\---

# 5\. OHLC

Understand:

* Open;
* High;
* Low;
* Close;
* Volume when available;
* timestamp;
* interval;
* trading session.

Every valid candle must satisfy:

High >= max(Open, Close)

Low <= min(Open, Close)

Understand that OHLC is a compressed representation of the price path.

OHLC does not reveal the exact intrabar sequence, so it cannot always establish whether a stop or target was reached first.

Resources:

* StockCharts ChartSchool: https://chartschool.stockcharts.com/
* CFI OHLC: https://corporatefinanceinstitute.com/resources/capital-markets/ohlc/

\---

# 6\. Candlesticks

Learn:

* body;
* upper wick;
* lower wick;
* bullish/bearish candle;
* total range;
* body/range ratio.

Useful derived quantities:

Body = |Close - Open|

Range = High - Low

Study the distribution of candle shapes rather than relying only on named candlestick patterns.

Resources:

* StockCharts ChartSchool: https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-analysis
* CFI Candlestick: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/candlestick/

\---

# 7\. Price Changes and Returns

Understand:

Delta P\_t = P\_t - P\_(t-1)

Simple return:

R\_t = P\_t/P\_(t-1) - 1

Percentage return:

100 R\_t

Log return:

r\_t = ln(P\_t/P\_(t-1))

Understand why returns are usually more useful than raw price changes when comparing different price levels.

Resources:

* Investopedia Rate of Return: https://www.investopedia.com/terms/r/rateofreturn.asp
* QuantStart Returns: https://www.quantstart.com/articles/Calculating-Returns/

\---

# 8\. Log Returns

Learn why:

r\_t = ln(P\_t/P\_(t-1))

is useful in time-series modelling.

Understand:

ln(P\_t/P\_(t-k)) = sum of the intervening log returns.

Compare simple and log returns and understand when their difference becomes important.

Resource:

* QuantStart: https://www.quantstart.com/articles/Calculating-Returns/

\---

# 9\. Mean and Standard Deviation

Understand:

x-bar = (1/n) sum(x\_i)

and sample standard deviation:

s = sqrt\[sum((x\_i-x-bar)^2)/(n-1)]

Learn:

* sample vs population variance;
* standard error;
* sensitivity to outliers;
* why standard deviation is used as a volatility measure.

Resource:

* NIST Handbook: https://www.itl.nist.gov/div898/handbook/

\---

# 10\. Rolling Statistics

Understand:

* rolling mean;
* rolling standard deviation;
* rolling minimum/maximum;
* rolling return;
* rolling volatility.

A 20-period statistic cannot be meaningfully calculated for the first 19 observations without special treatment.

Do not replace warm-up values with arbitrary numbers.

Resource:

* pandas rolling: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

\---

# 11\. Volatility

Understand volatility as variability, not direction.

Learn:

* realized volatility;
* rolling volatility;
* return standard deviation;
* intraday vs daily volatility;
* annualisation;
* volatility clustering.

Understand the approximate scaling:

sigma\_annual = sigma\_period \* sqrt(N)

only when the assumptions behind the scaling are appropriate.

Resources:

* Investopedia Volatility: https://www.investopedia.com/terms/v/volatility.asp
* NIST: https://www.itl.nist.gov/div898/handbook/

\---

# 12\. Confidence Intervals

The current project uses a rolling 95% t-interval for a sample mean.

Understand:

CI = x-bar +/- t\* s/sqrt(n)

Learn:

* confidence level;
* critical value;
* standard error;
* degrees of freedom;
* t distribution.

Most importantly:

A confidence interval for the mean is **not** a 95% probability range for the next price or return.

Investigate how autocorrelation and non-independent observations affect naive confidence intervals.

Resources:

* NIST: https://www.itl.nist.gov/div898/handbook/prc/section2/prc221.htm
* Penn State: https://online.stat.psu.edu/stat500/lesson/confidence-intervals

\---

# 13\. Simple Moving Average

Understand:

SMA\_n = (1/n) sum(P\_i)

The project currently uses 5- and 12-period SMAs.

Learn:

* smoothing;
* lag;
* window size;
* first valid observation;
* sensitivity to the selected period.

Resources:

* StockCharts: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential
* CFI: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/moving-average/

\---

# 14\. Moving-Average Crossovers

Understand how the 5/12 crossover is detected.

Learn:

* trend indication;
* lag;
* false signals;
* sideways-market behaviour;
* parameter sensitivity;
* look-ahead bias.

A crossover count is descriptive. It is not automatically a profitable trading strategy.

Resource:

* StockCharts: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential

\---

# 15\. Bollinger Bands

Understand:

Middle = SMA\_n

Upper = SMA\_n + k sigma

Lower = SMA\_n - k sigma

Current project:

n = 20

k = 2

Learn:

* rolling mean;
* rolling standard deviation;
* band width;
* price position;
* expansion;
* contraction.

Do not interpret touching an upper/lower band as an automatic buy/sell instruction.

Resources:

* StockCharts: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands
* Official Bollinger Bands: https://www.bollingerbands.com/

\---

# 16\. True Range and ATR

True Range:

TR\_t = max(
High\_t - Low\_t,
abs(High\_t - Close\_(t-1)),
abs(Low\_t - Close\_(t-1))
)

ATR is a rolling average of True Range.

Learn why True Range incorporates the previous close and therefore captures gaps.

Understand:

* ATR measures range/volatility;
* ATR does not indicate direction;
* increasing ATR means larger typical ranges;
* decreasing ATR means smaller typical ranges.

Resource:

* StockCharts ATR: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr

\---

# 17\. RSI

Current project: 14-period RSI.

Understand:

RS = Average Gain / Average Loss

RSI = 100 - 100/(1+RS)

Learn:

* gains/losses;
* smoothing;
* 0-100 scale;
* momentum;
* conventional 30/70 reference levels;
* divergence;
* trend-dependent interpretation.

Do not interpret RSI > 70 as proof that price must fall or RSI < 30 as proof that price must rise.

Resources:

* StockCharts RSI: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi
* Optional Bloomberg RSI material: https://www.bloomberg.com/professional/insights/webinar/charts-of-the-month-what-is-the-rsi-and-how-to-use-it-in-trading-strategies/

Bloomberg is supplementary only.

\---

# 18\. MACD

Current project:

MACD = EMA\_12 - EMA\_26

Signal = EMA\_9(MACD)

Histogram = MACD - Signal

Learn:

* EMA;
* SMA vs EMA;
* fast/slow averages;
* zero-line crossing;
* signal crossover;
* histogram;
* momentum;
* lag.

Understand why MACD and RSI can provide different information.

Resource:

* StockCharts MACD: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator

\---

# 19\. Drawdown

Understand:

Drawdown = current value - running maximum

Percentage drawdown:

DD% = (Value - Running Maximum)/Running Maximum

Learn:

* running maximum;
* peak;
* trough;
* recovery;
* maximum drawdown;
* duration of drawdown.

Drawdown is not the same thing as volatility.

Resources:

* Investopedia: https://www.investopedia.com/terms/m/mdd.asp
* CFA Institute: https://www.cfainstitute.org/

\---

# 20\. Sharpe Ratio

Understand:

Sharpe = (R\_p - R\_f)/sigma\_p

For the project's existing implementation, document exactly what is used for:

* return;
* risk-free rate;
* volatility;
* time period.

Do not automatically annualise a per-bar ratio.

Learn:

* excess return;
* volatility as a risk proxy;
* sampling error;
* non-normal returns;
* serial dependence.

Resources:

* CFA Institute: https://rpc.cfainstitute.org/research/foundation/2019/understanding-the-sharpe-ratio
* Investopedia: https://www.investopedia.com/terms/s/sharperatio.asp

\---

# 21\. Sortino Ratio

Understand the difference from Sharpe.

Sharpe penalises overall volatility.

Sortino focuses on downside deviation relative to a target/minimum acceptable return.

Learn:

* target return;
* downside deviation;
* negative deviations;
* implementation choices.

Document the exact implementation used in the project.

Resource:

* Investopedia: https://www.investopedia.com/terms/s/sortinoratio.asp

\---

# 22\. Risk/Reward and R-Multiples

Understand:

R = initial risk

A 2R target means a reward equal to twice the initial risk.

Current project:

* entry = next bar's open;
* stop = signal candle low/high;
* target = 2R;
* if stop and target occur in one candle, stop is assumed to occur first.

This last rule is a modelling convention because OHLC data cannot establish intrabar ordering.

Resource:

* Investopedia: https://www.investopedia.com/articles/trading/09/risk-reward-ratio.asp

\---

# 23\. Historical Trade Analysis

The existing project selects the best historical crossover trade using later bars.

This is hindsight analysis.

Learn:

* look-ahead bias;
* in-sample analysis;
* out-of-sample testing;
* data snooping;
* overfitting.

Never describe the best historical trade as a forecast.

\---

# 24\. Statistical Dependence

Investigate:

* autocorrelation of returns;
* autocorrelation of absolute returns;
* autocorrelation of squared returns;
* volatility persistence.

A key modelling question is whether simulated data reproduces dependence observed in real data.

Resources:

* NIST: https://www.itl.nist.gov/div898/handbook/
* statsmodels ACF: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html

\---

# 25\. Return Distributions

Compare observed and simulated:

* mean;
* median;
* standard deviation;
* skewness;
* kurtosis;
* quantiles;
* tails.

Use:

* histograms;
* density plots;
* ECDFs;
* Q-Q plots.

Do not assume matching mean and standard deviation means the simulation is realistic.

Resources:

* NIST: https://www.itl.nist.gov/div898/handbook/
* SciPy statistics: https://docs.scipy.org/doc/scipy/reference/stats.html

\---

# 26\. Empirical Calibration

The long-term pipeline should be:

Observed data
-> clean data
-> calculate returns/ranges
-> estimate empirical properties
-> calibrate simulation
-> generate simulated OHLC
-> analyse
-> compare observed vs simulated.

Potential calibration variables:

* return mean;
* return standard deviation;
* return quantiles;
* high-low range;
* candle body/range;
* opening gaps;
* volatility persistence;
* autocorrelation.

A calibrated simulation reproduces selected statistical properties. It does not automatically predict future prices.

\---

# 27\. Coinbase Integration

Coinbase should be the first external real-market source.

Start with BTC-USD.

Use historical Coinbase candles to calibrate the simulation.

Use Coinbase live data later to initialise or demonstrate a real-time data stream.

Do not use Coinbase for:

* FTSE 100 equities;
* conventional FX.

Create provider-specific adapters with one common normalized data format.

Official resources:

* Advanced Trade API: https://docs.cdp.coinbase.com/advanced-trade/
* WebSocket overview: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview/
* Public product candles: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/public/get-public-product-candles
* Public market trades: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/public/get-public-market-trades
* Python SDK: https://github.com/coinbase/coinbase-advanced-py

\---

# 28\. Coinbase Historical Data

Normalize downloaded data into:

* timestamp;
* open;
* high;
* low;
* close;
* volume;
* symbol;
* asset class;
* currency;
* source;
* data type.

For example:

symbol = BTC-USD
asset\_class = crypto
source = coinbase
data\_type = observed

Record:

* download time;
* requested interval;
* start/end time;
* symbol;
* provider;
* observation count.

Keep raw data separate from transformed data.

\---

# 29\. Coinbase Live Data

Use REST for historical calibration.

Use WebSocket for live streaming.

Understand the difference between:

* latest price;
* trade events;
* candles;
* order-book data.

If trades are aggregated into candles, explicitly define:

* interval;
* open;
* high;
* low;
* close;
* volume.

Handle:

* reconnects;
* missing messages;
* duplicate events;
* timestamp ordering;
* network failures.

A live price is an initial condition, not a forecast.

\---

# 30\. FX Mode

Potential currencies:

* GBP;
* USD;
* EUR;
* JPY;
* AUD;
* NZD.

Store base and quote currency separately.

Define BASE/QUOTE as quote-currency units per one base-currency unit.

Verify the selected provider's symbol convention.

Consider:

* weekends;
* operating hours;
* timezones;
* intraday intervals.

Do not use Coinbase as the FX provider.

\---

# 31\. FTSE 100 Mode

Begin with ten constituents.

Store:

* company;
* ticker;
* exchange;
* currency;
* provider;
* quote timestamp.

Keep the constituent list separate from simulation logic.

Verify whether data is:

* real-time;
* delayed;
* end-of-day.

Use an appropriate equity-market data provider.

Do not use Coinbase for equities.

\---

# 32\. Provider Architecture

The simulation should request a generic function such as:

get\_historical\_ohlcv()

rather than directly depending on Coinbase-specific JSON.

Potential provider capabilities:

* get\_historical\_ohlcv();
* get\_latest\_price();
* stream\_trades();
* stream\_market\_data().

Implement in stages:

1. Coinbase.
2. FX provider.
3. Equity provider.

The downstream Python/R analysis should not care which provider produced the data.

\---

# 33\. Timeframes

Support:

* 1d;
* 5d;
* 1m;
* 3m;
* 1y;
* ytd.

Normalize input.

Reject invalid choices.

Use calendar-aware date calculations.

Do not treat a month as exactly 30 days.

Keep historical timeframe and future simulation horizon conceptually separate.

YTD means January 1 through today.

If a full January-December simulation is required, name it separately.

\---

# 34\. Sessions and Bars

Decide whether "days" means calendar days or trading sessions.

For equities, use market sessions.

For crypto, recognise that trading is continuous.

For FX, account for its different market structure.

Generate timestamps separately for each asset class.

Ensure:

* timestamps increase;
* timestamps are unique;
* bars fall inside intended sessions;
* actual bar count is derived from timestamps.

\---

# 35\. OHLC Generator Improvements

Keep the current random model as a baseline.

Refactor the generator so that:

* starting price is explicit;
* timestamps are explicit;
* volatility is explicit;
* price-floor behaviour is explicit;
* random seed is optional;
* module-level mutable state is avoided.

Generate close-to-close returns separately from intrabar range.

Never generate High and Low independently without enforcing candle structure.

Test:

High >= max(Open, Close)

Low <= min(Open, Close)

\---

# 36\. Data Quality

Before analysis, check:

* missing values;
* duplicate timestamps;
* invalid OHLC;
* non-finite values;
* unexpected gaps;
* timezone errors;
* incorrect symbol;
* malformed provider responses.

Do not silently substitute failed real data with random data.

Log data-quality decisions.

\---

# 37\. API Security

Never store API credentials in:

* source files;
* CSVs;
* PNG metadata;
* README files;
* Git history.

Use environment variables or ignored local configuration.

Maintain a no-network test mode using saved API responses.

\---

# 38\. Python

Python should handle:

* API access;
* simulation;
* CSV creation;
* PNG creation;
* core numerical analysis.

Potential packages:

* pandas;
* numpy;
* scipy;
* matplotlib;
* requests;
* Coinbase's official SDK where appropriate.

Official resources:

* NumPy: https://numpy.org/doc/
* pandas: https://pandas.pydata.org/docs/
* SciPy: https://docs.scipy.org/doc/scipy/
* Matplotlib: https://matplotlib.org/stable/

\---

# 39\. R

R should handle statistical exploration and visualisation where useful.

Potential packages:

* tidyverse;
* ggplot2;
* dplyr;
* readr;
* lubridate;
* zoo;
* xts;
* PerformanceAnalytics.

Do not add packages without a reason.

The project should clearly document which calculations are performed by Python and which by R.

Resources:

* R for Data Science: https://r4ds.hadley.nz/
* tidyverse: https://www.tidyverse.org/
* ggplot2: https://ggplot2.tidyverse.org/

\---

# 40\. PNG Interpretation Checklist

### Price

* What is the trend?
* What are typical candle sizes?
* Are ranges expanding?
* Are there gaps?

### Moving averages

* Where are the averages relative to each other?
* Where are the crossovers?
* How much lag is present?

### Bollinger Bands

* Is volatility expanding or contracting?
* Where is price relative to the rolling mean?
* Is band width unusual?

### RSI

* What does recent momentum look like?
* Is RSI near a conventional threshold?
* Is the market trending?

### MACD

* Is momentum increasing/decreasing?
* Is MACD above/below zero?
* Is the histogram expanding/contracting?

### ATR

* Are typical ranges increasing or decreasing?

### Drawdown

* How far is price below the previous peak?
* How long is recovery taking?

### Risk metrics

* How large are returns relative to volatility?
* Is downside risk large?
* Is the sample large enough to interpret the ratio?

The PNG should be interpreted as a set of interacting measurements, not as a collection of automatic trading signals.

\---

# 41\. Conflicting Indicators

Indicators measure different properties.

For example:

* RSI -> recent momentum;
* MACD -> moving-average momentum;
* ATR -> range/volatility;
* Bollinger Bands -> rolling location and dispersion;
* drawdown -> distance from a previous peak.

Therefore:

RSI high + MACD positive + ATR increasing + price below a previous peak

is not necessarily contradictory.

Learn why each metric can describe a different dimension of the same market state.

\---

# 42\. Descriptive, Inferential and Predictive Analysis

### Descriptive

"What happened in this dataset?"

Examples:

* average return;
* bullish-candle percentage;
* ATR;
* maximum drawdown.

### Inferential

"What can this sample tell us about a wider process, subject to assumptions?"

Examples:

* confidence intervals;
* statistical tests;
* distribution estimation.

### Predictive

"What might happen next?"

The project should not label an indicator or simulation as predictive without a properly tested predictive model.

A simulation can generate plausible paths without predicting which path will occur.

\---

# 43\. Model Validation

Compare observed and simulated data using:

* return mean;
* standard deviation;
* quantiles;
* skewness;
* kurtosis;
* high-low ranges;
* candle-body distributions;
* volatility clustering;
* drawdown distributions;
* indicator distributions.

Use both numerical and visual comparisons.

Useful diagnostics:

* histogram;
* ECDF;
* Q-Q plot;
* rolling volatility;
* autocorrelation;
* drawdown curves.

\---

# 44\. Overfitting

Do not optimise a simulation until it perfectly matches one historical period.

Separate:

* calibration data;
* validation data.

Example:

Historical calibration period
-> estimate parameters
-> generate simulations
-> compare against validation period.

Learn:

* overfitting;
* data snooping;
* parameter sensitivity;
* out-of-sample validation.

\---

# 45\. Output Compatibility

Preserve:

run\_X.csv

run\_X.png

Every CSV should have a matching PNG.

Where useful, include:

* run ID;
* symbol;
* market type;
* timeframe;
* source;
* data type;
* timezone;
* calibration window;
* random seed.

Cleanup code must never remove source data, notes, tests or configuration.

\---

# 46\. Testing

Test:

* timeframe parsing;
* date boundaries;
* leap years;
* sessions;
* OHLC invariants;
* zero volatility;
* low/high starting prices;
* missing data;
* duplicate timestamps;
* malformed API responses;
* network errors;
* rate limits;
* deterministic seeds;
* CSV/PNG matching;
* indicator warm-up;
* each provider independently.

The project must remain testable without internet access.

\---

# OPTION 1 — HISTORICAL BACKTEST + SIMULATION ACCURACY INDEX

## 47\. Purpose

Add an optional mode where the user can select:

* market;
* symbol;
* candle interval;
* historical start date;
* historical end date;
* number of simulations;
* random seed or seed range.

The existing Python simulator already supports repeated simulations.

The new layer should retrieve **observed historical OHLCV data** for exactly the same date/candle range and compare each simulated CSV against the observed CSV.

The CSV should be the primary source for the calculation.

The PNG should then display the resulting score.

The purpose is to answer:

> "How closely did this simulation reproduce the observed market path over this historical interval?"

It must not be described as proof that the model predicts future prices.

\---

## 48\. Historical Data Source

Do not make TradingView the only source.

Use a provider hierarchy appropriate to the asset:

### Crypto

Coinbase Advanced Trade is a strong first source for BTC-USD because it provides historical market-data endpoints and real-time WebSocket market data.

Official documentation:

https://docs.cdp.coinbase.com/advanced-trade/

### Equities

Investigate providers such as:

* Twelve Data;
* Polygon/Massive;
* another provider with explicit historical OHLCV coverage.

Twelve Data documents API endpoints for time series and other market datasets:

https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request

Polygon/Massive documentation should be checked for the specific asset, exchange and historical entitlement before implementation.

### FX

Use a provider that explicitly supports historical FX candles and volume/tick-volume fields where available.

Important:

FX "volume" is provider-dependent. Spot FX does not have one centralized exchange volume number comparable to an exchange-traded stock or crypto venue. If the provider supplies tick volume, label it as tick volume rather than pretending it is centralized traded volume.

\---

## 49\. TradingView's Role

TradingView may be useful as a visual cross-check of the selected historical candle.

Do not build the accuracy engine around scraping TradingView pages.

Prefer:

1. official provider APIs;
2. exchange/vendor data;
3. TradingView as a visual/reference check where appropriate.

The project should store the provider and exact data source used for every backtest.

This prevents a "TradingView says..." dependency from becoming embedded in the numerical model.

\---

## 50\. Backtest Input

Example configuration:

```text
mode = backtest
symbol = BTC-USD
interval = 1h
start = 2025-01-01
end = 2025-01-07
simulations = 100
seed = 42

