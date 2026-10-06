import yfinance as yf

def get_starting_price(symbol):
    try:
        # Download the most recent 1-day interval data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d")
        
        if df.empty:
            return f"No data found for symbol: {symbol}"
        
        # Pull the Daily Open and the Last Close/Current price
        open_price = df['Open'].iloc[0]
        current_price = df['Close'].iloc[0]
        
        return {
            "symbol": symbol,
            "open": round(open_price, 7),
            "current": round(current_price, 7)
        }
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # 1. Stocks (Use standard ticker)
    stock_data = get_starting_price("AAPL")
    if isinstance(stock_data, dict):
        print(f"Stock ({stock_data['symbol']}) -> Open: ${stock_data['open']} | Current: ${stock_data['current']}")
    else:
        print(f"Stock -> {stock_data}")

    # 2. FX Pairs (Format: BASECURRENCY+TERMCURRENCY=X)
    fx_data = get_starting_price("GBPUSD=X")
    if isinstance(fx_data, dict):
        print(f"FX ({fx_data['symbol']})    -> Open: {fx_data['open']} | Current: {fx_data['current']}")
    else:
        print(f"FX -> {fx_data}")

    # 3. Crypto (Format: TOKEN-CURRENCY, e.g., BTC-USD)
    crypto_data = get_starting_price("BTC-USD")
    if isinstance(crypto_data, dict):
        print(f"Crypto ({crypto_data['symbol']}) -> Open: ${crypto_data['open']} | Current: ${crypto_data['current']}")
    else:
        print(f"Crypto -> {crypto_data}")
