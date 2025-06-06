from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier
import os
from datetime import datetime
import pytz

app = Flask(__name__)

# === CONFIG ===
DATA_DIR = "SP500_data"
os.makedirs(DATA_DIR, exist_ok=True)

# === SYMBOL FIX ===
def fix_symbol(symbol):
    return symbol.replace('.', '-')

# === CHECK IF MARKET OPEN ===
def is_market_open():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    eastern = pytz.timezone('US/Eastern')
    now_est = now_utc.astimezone(eastern)
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    return market_open <= now_est.time() <= market_close

# === FETCH DATA ===
def fetch_data(symbol):
    symbol = fix_symbol(symbol)
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    def download():
        df = yf.Ticker(symbol).history(period="300d")
        if df.empty:
            raise ValueError(f"Brak danych dla symbolu '{symbol.upper()}'. Sprawdź, czy symbol jest poprawny.")
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(file_path, index=False)
        return df

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if df.empty:
            return download()

        last_date = df['Date'].max().date()
        today = datetime.today().date()
        if (today - last_date).days > 5:
            return download()
        if last_date >= today or not is_market_open():
            return df
        new_data = yf.Ticker(symbol).history(start=last_date + pd.Timedelta(days=1))
        if not new_data.empty:
            new_data.reset_index(inplace=True)
            new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = pd.concat([df, new_data])
            df.drop_duplicates('Date', keep='last', inplace=True)
            df.sort_values('Date', inplace=True)
            df.to_csv(file_path, index=False)
        return df
    return download()

# === TRAIN MODELS ===
def train_models(df, gap_threshold=1.0):
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['GapTarget'] = (df['Gap'].abs() > gap_threshold / 100).astype(int)
    df['NextCandle'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    gap_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.08, verbose=False)
    gap_model.fit(X, df['GapTarget'])

    candle_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.08, verbose=False)
    candle_model.fit(X, df['NextCandle'])

    return gap_model, candle_model

# === ROUTES ===
@app.route("/tickers")
def tickers():
    return send_from_directory("static/data", "sp500.json")

@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.json.get("symbol")
    threshold = float(request.json.get("threshold", 1.0))
    try:
        df = fetch_data(symbol)
        gap_model, candle_model = train_models(df, gap_threshold=threshold)
        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]

        gap_proba = gap_model.predict_proba(latest)[0][1]
        candle_proba = candle_model.predict_proba(latest)[0][1]
        candle_color = "white" if candle_proba > 0.5 else "black"

        df_recent = df.tail(60).copy()
        df_recent['Date'] = df_recent['Date'].dt.strftime('%Y-%m-%d')
        ohlc = df_recent.to_dict(orient='records')

        return jsonify({
            "gap_probability": f"{gap_proba * 100:.2f}%",
            "candle_prediction": candle_color,
            "candle_probability": f"{candle_proba * 100:.2f}%",
            "ohlc": ohlc
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
