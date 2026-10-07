# Plan: Price Modeling and Market Modes

## Purpose

- Extend the existing price simulation script in small, testable steps.
- Keep the current random-market simulation available as one of three modes.
- Add timeframe selection for all supported modes.
- Improve simulated OHLC bars using empirical market data where possible.
- Add a foreign-exchange mode and a FTSE 100 company mode.
- Keep the analysis pipeline compatible with generated CSV and PNG pairs.
- Implement the changes manually; do not use generative AI to write the code.
- Treat generated future prices as simulations, not forecasts or financial advice.

## Current Code Map

- `price_modeling.py` currently reads inputs at module level.
- `generate_daily()` creates one sequence of Open, High, Low, and Close values.
- `all_timestamps` currently represents a fixed five-day period.
- Each run writes one numbered CSV and one numbered PNG.
- The script calls `analyze_looped_results()` after generating the run pairs.
- `market_analysis.py` validates and analyzes the CSV/PNG pairs.

## Decisions Before Coding

- Decide whether each horizon means historical calendar time or future simulated time.
- Keep these concepts distinct: downloaded observations are historical/real-time data.
- Future bars generated from those observations are simulated data.
- Define whether `1d` means the rest of today or the next complete trading session.
- Define whether a market holiday shifts the first simulated session forward.
- Define the timezone used for each market's session timestamps.
- Use UTC internally where practical, then convert for market session display.
- Treat YTD as January 1 through the current date by the standard definition.
- If the goal is January 1 through December 31, name it `1y` or `calendar year`.
- Decide how to handle YTD when the current date is near year-end.
- Decide whether each looped run reuses one market snapshot or fetches a new one.
- Decide if the ten FTSE companies are all generated in one run or selected individually.

## Milestone 1: Timeframe Selection

- Add one timeframe input with choices `1d`, `5d`, `1m`, `3m`, `1y`, and `ytd`.
- Normalize input by trimming spaces and converting it to lowercase.
- Reject unknown choices with a clear message and reprompt or exit cleanly.
- Use `datetime.date.today()` or an equivalent current-date API.
- Do not hard-code the current year or a fixed starting date.
- Use calendar-aware date offsets for months and years.
- Do not approximate one month as a fixed 30-day timedelta.
- Define an inclusive start and end date for every timeframe.
- For future simulation, define the first simulated date and the horizon end explicitly.
- For YTD history, use January 1 of the current year through today.
- For a full calendar-year simulation, use a separate full-year option or clear label.
- Generate the date range in a dedicated helper function.
- Keep date-range selection separate from OHLC generation.
- Return dates and timeframe metadata from the helper.
- Add tests for a month ending on the 30th and on the 31st.
- Add a leap-year test for February and the one-year horizon.
- Add a year-boundary test for YTD.
- Verify that invalid inputs do not delete existing outputs.

## Milestone 2: Sessions and Bar Count

- Decide whether the requested `9 * days` means calendar days or trading sessions.
- Prefer nine bars per market session for stock charts.
- Generate session dates separately for FX and UK-listed equities if their calendars differ.
- A weekday-only calendar is a first approximation, not a holiday calendar.
- Add a market calendar later if public holidays need to be accurate.
- For FX, define what nine daily observations mean across a 24-hour weekday market.
- Generate nine timestamps per selected session after the session set is finalized.
- Derive bar count from actual generated timestamps, not a separate formula.
- Check that timestamps are increasing and unique.
- Check that bars fall inside the intended date range and session hours.
- Label chart axes and titles with the selected timeframe and timezone.
- Record the actual first and last bar timestamps in CSV metadata or summary output.

## Milestone 3: Validate the Existing OHLC Generator

- Keep `generate_daily()` focused on producing bars from explicit inputs.
- Pass the starting price, timestamps, volatility, and price-floor setting as arguments.
- Avoid hidden dependencies on mutable module-level variables.
- Preserve the current random simulation as mode 3 until other modes are stable.
- Use a local random generator with an optional seed for repeatable tests.
- Model close-to-close returns separately from intrabar High/Low ranges.
- Do not choose High and Low using independent unrestricted random changes.
- Ensure `High >= max(Open, Close)` for every generated bar.
- Ensure `Low <= min(Open, Close)` for every generated bar.
- Ensure all generated OHLC values are finite numbers.
- Apply the nonnegative-price floor only when the selected instrument requires it.
- Allow negative prices only for instruments where that behavior is explicitly justified.
- Add tests for zero volatility, a low starting price, and a high starting price.
- Add tests that validate every candle's OHLC ordering.

## Milestone 4: Research Empirical High/Low Behavior

- Choose a data provider before implementing a network request.
- Confirm that the provider supports the requested asset class and interval.
- Confirm historical intraday OHLC availability, not only a latest quote endpoint.
- Review API documentation manually and record the exact endpoint and fields used.
- Check licensing, attribution, account requirements, and redistribution limits.
- Check request limits and design for throttling rather than repeated rapid requests.
- Keep API keys out of source files and out of generated CSVs.
- Load a small sample of historical bars and inspect it before modeling from it.
- Calculate empirical returns and high-low ranges from the sample.
- Consider using normalized ranges so behavior scales with price level.
- Consider separating opening gaps from intrabar price movement.
- Compare generated ranges with observed ranges using summary statistics.
- Keep historical observations clearly distinguished from simulated observations.
- Do not imply that matching historical ranges predicts future prices.

## Milestone 5: Three-Mode Input Flow

- Add a top-level market-type choice with exactly three documented options.
- Option 1 selects FX.
- Option 2 selects a FTSE 100 company or company batch.
- Option 3 selects the existing random simulation.
- Validate the selected option before clearing or writing any output files.
- Keep mode-specific prompts in small functions rather than one large input block.
- Return a common configuration object or clearly defined values from each prompt.
- Include market type, symbol, timeframe, and quote currency in output metadata.
- Give invalid choices a clear error and avoid partially generated output.
- Test each mode's configuration without making a network request.

## Milestone 6: FX Pair Selection

- Use ISO 4217 currency codes for choices: JPY, EUR, USD, AUD, NZD, and GBP.
- Display readable labels while storing stable codes internally.
- Ask for base currency and quote currency as two separate validated choices.
- Reject selecting the same currency twice unless there is a specific reason to allow it.
- Define pair notation explicitly as `BASE/QUOTE`.
- Explain that a `BASE/QUOTE` value is quote-currency units per one base-currency unit.
- Confirm the provider's symbol convention before assembling an FX symbol.
- Do not assume that all providers use the same direction or delimiter.
- If a provider only returns the inverse pair, invert rates carefully and document it.
- Use actual FX data from a provider that supports FX; Coinbase is crypto-focused.
- A latest quote can initialize a simulation but cannot supply realistic OHLC history.
- Use historical intraday FX candles to calibrate returns and intrabar ranges.
- Account for FX operating hours, weekends, and the selected provider's timezone.

## Milestone 7: FTSE 100 Company Selection

- Choose ten FTSE 100 constituents and document why those ten were selected.
- Decide whether the user chooses one constituent or generates all ten in a batch.
- Store display names separately from provider-specific ticker symbols.
- Verify each ticker and exchange suffix using the selected provider's documentation.
- Confirm the provider supplies UK market data for those symbols.
- Check whether the data is delayed, real-time, or end-of-day.
- Use the London Stock Exchange trading calendar and timezone when available.
- Handle market closures and non-trading dates explicitly.
- Do not use Coinbase for ordinary listed-company equities.
- Keep the constituent list easy to update because index membership changes.
- Record the symbol, company name, currency, provider, and quote timestamp.

## Milestone 8: Provider and Network Reliability

- Implement one small provider adapter at a time.
- Keep HTTP requests separate from simulation and chart code.
- Set connection and response timeouts.
- Check HTTP status codes and validate response fields before using values.
- Handle empty responses, malformed JSON, rate limits, and unavailable symbols.
- Avoid silently replacing failed real data with random data.
- Cache a successful quote or historical response only if its age is recorded.
- Print a clear message when network data is unavailable.
- Do not commit secrets, API keys, tokens, or account-specific configuration.
- Use environment variables or a local ignored configuration for credentials.
- Provide a no-network test path using saved sample responses.

## Milestone 9: Output and Analysis Compatibility

- Preserve the paired numbered naming convention for source CSV and PNG files.
- Ensure every source CSV has a matching source PNG with the same run number.
- Consider adding instrument and timeframe columns to CSV output.
- Keep CSV headers normalized with no leading or trailing whitespace.
- Preserve the market-analysis learning guide in the analysis output directory.
- Verify that `market_analysis.py` accepts the generated timeframe length.
- Review any analysis indicators whose periods exceed a one-day bar count.
- Make one-day charts valid when some long-period indicators are unavailable.
- Keep generated output cleanup limited to known generated files.
- Avoid deleting notes, source data, or unrelated user files during cleanup.

## Suggested Manual Work Order

- Implement the timeframe helper and test its calendar boundaries first.
- Generate market-session timestamps and pass them into the OHLC generator.
- Validate OHLC invariants and compare ranges with a saved historical sample.
- Add the modes and provider adapters incrementally, starting with one symbol.
- Run an end-to-end test in temporary output folders before using live data.
- Review credential handling and output cleanup before using network APIs.

## Acceptance Checks

- Every timeframe option maps to the documented start and end dates.
- The generated bar count matches the session calendar and bars-per-session rule.
- YTD behavior is documented and is not confused with a future full-year simulation.
- Every candle satisfies the High/Low relationship to Open and Close.
- Every CSV and PNG pair has matching instrument and run identifiers.
- Invalid input and provider errors do not erase prior outputs before validation.
- FX symbols use the selected provider's verified pair convention.
- The ten equity symbols resolve to the intended FTSE constituents.
- The simulation is labeled as simulated and does not claim to predict prices.
- The existing analysis pipeline completes on sample output for each mode.
- Tests use fixtures or temporary directories and do not overwrite user results.
- The learning-guide Markdown file is copied alongside generated analysis outputs.
