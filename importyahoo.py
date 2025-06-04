import yfinance as yf

ticker = yf.Ticker("AAPL")
df = ticker.history(period="300d")

print(df.head())
