import yfinance as yf
import pandas as pd

# Pobierz listę spółek S&P 500 z Wikipedii
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
df = tables[0]
df = df[['Security', 'Symbol']]

# Funkcja do pobrania kapitalizacji rynkowej
def get_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('marketCap', 0)
    except:
        return 0

# Dodaj kapitalizację rynkową do DataFrame
df['MarketCap'] = df['Symbol'].apply(get_market_cap)

# Posortuj według kapitalizacji rynkowej
df_sorted = df.sort_values(by='MarketCap', ascending=False).reset_index(drop=True)

# Przygotuj słownik {nazwa_spółki: ticker}
sp500_full_sorted = dict(zip(df_sorted['Security'], df_sorted['Symbol']))

# Wyświetl pierwsze 10 spółek
print(sp500_full_sorted)
