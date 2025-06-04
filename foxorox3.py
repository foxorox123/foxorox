import pandas as pd
import tkinter as tk
from tkinter import ttk
from catboost import CatBoostClassifier
from datetime import datetime
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
import yfinance as yf
import time

# 10 największych spółek S&P 500 i ich symbole
sp500_top10 = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'NVIDIA': 'NVDA',
    'Alphabet (GOOGL)': 'GOOGL',
    'Alphabet (GOOG)': 'GOOG',
    'Berkshire Hathaway': 'BRK-B',
    'Meta Platforms': 'META',
    'Tesla': 'TSLA',
    'UnitedHealth Group': 'UNH'
}

def fetch_data(symbol):
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="300d")
            if df.empty:
                print(f"No data for {symbol}")
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            print(f"Failed to get ticker '{symbol}', attempt {attempt+1}: {e}")
            time.sleep(2)
    return pd.DataFrame()

def train_gap_model(df):
    df = df.copy()
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['GapAboveThreshold'] = (df['Gap'].abs() > threshold.get() / 100).astype(int)
    df = df.dropna()
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['GapAboveThreshold']
    if len(y.unique()) == 1:
        raise ValueError("Only one unique value in target")
    model = CatBoostClassifier(
        iterations=iterations.get(),
        depth=depth.get(),
        learning_rate=learning_rate.get(),
        verbose=False
    )
    model.fit(X, y)
    return model

def train_candle_model(df):
    df = df.copy()
    df['NextCandleWhite'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    df = df.dropna()
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['NextCandleWhite']
    if len(y.unique()) == 1:
        raise ValueError("Only one unique value in target")
    model = CatBoostClassifier(
        iterations=iterations.get(),
        depth=depth.get(),
        learning_rate=learning_rate.get(),
        verbose=False
    )
    model.fit(X, y)
    return model

def plot_candlestick(ax, data):
    width = 0.6
    colors_up = 'green'
    colors_down = 'red'
    for idx, row in data.iterrows():
        date_num = mdates.date2num(row['Date'])
        open_, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        color = colors_up if close >= open_ else colors_down
        rect = Rectangle((date_num - width/2, min(open_, close)), width, abs(close - open_), color=color)
        ax.add_patch(rect)
        ax.plot([date_num, date_num], [low, high], color=color, linewidth=1)

def show_plot(df, title):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(7, 6), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    data_tail = df.tail(50).copy()
    data_tail['Volume'] = data_tail['Volume'] / 1_000_000  # mln shares

    if plot_type.get() == "Line":
        ax1.plot(data_tail['Date'], data_tail['Close'], label="Close", color='blue')
    else:
        plot_candlestick(ax1, data_tail)

    ax1.set_title(title)
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    ax2.bar(data_tail['Date'], data_tail['Volume'], color='gray', width=0.6)
    ax2.set_ylabel("Volume (mln shares)")
    ax2.grid(True)

    fig.autofmt_xdate(rotation=45)
    fig.subplots_adjust(hspace=0.1)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def update_prediction():
    try:
        symbol = sp500_top10[company_var.get()]
        df = fetch_data(symbol)
        if df.empty:
            result_var.set("No data available")
            return

        gap_model = train_gap_model(df)
        candle_model = train_candle_model(df)
        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]

        gap_proba = gap_model.predict_proba(latest)[0][1]
        candle_proba = candle_model.predict_proba(latest)[0][1]
        candle_color = "white (bullish)" if candle_proba > 0.5 else "black (bearish)"

        result_var.set(
            f"{company_var.get()}:\n"
            f" - Chance for gap > {threshold.get()}%: {gap_proba*100:.2f}%\n"
            f" - Next candle predicted: {candle_color}\n"
            f" - Probability: {candle_proba*100:.2f}%"
        )
        show_plot(df, f"Chart: {company_var.get()}")
    except Exception as e:
        result_var.set("❌ Analysis error.")
        with open("errors.log", "a") as f:
            f.write(f"[{datetime.now()}] {type(e).__name__}: {str(e)}\n")

def calculate_all():
    for row in tree.get_children():
        tree.delete(row)

    results = []

    for company, symbol in sp500_top10.items():
        df = fetch_data(symbol)
        if df.empty:
            results.append((company, None, None, None, df))
            continue
        try:
            gap_model = train_gap_model(df)
            candle_model = train_candle_model(df)
        except Exception:
            results.append((company, None, None, None, df))
            continue

        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]

        gap_proba = gap_model.predict_proba(latest)[0][1]
        candle_proba = candle_model.predict_proba(latest)[0][1]
        candle_color = "white" if candle_proba > 0.5 else "black"

        results.append((company, gap_proba*100, candle_color, candle_proba*100, df))

    results.sort(key=lambda x: (x[3] is None, -x[3] if x[3] is not None else 0))

    for company, gap_pct, candle_col, candle_pct, _ in results:
        if gap_pct is None:
            tree.insert("", "end", values=(company, "No data", "-", "-"))
        else:
            tree.insert("", "end", values=(company, f"{gap_pct:.2f}%", candle_col, f"{candle_pct:.2f}%"))

    for company, gap_pct, candle_col, candle_pct, df in results:
        if candle_pct is not None:
            show_plot(df, f"Chart: {company} (highest probability)")
            break

def on_plot_type_change():
    update_prediction()

# === GUI ===
root = tk.Tk()
root.title("Foxorox")
root.geometry("1200x700")

threshold = tk.DoubleVar(value=1.0)
iterations = tk.IntVar(value=100)
depth = tk.IntVar(value=6)
learning_rate = tk.DoubleVar(value=0.1)
company_var = tk.StringVar(value=list(sp500_top10.keys())[0])
result_var = tk.StringVar()
plot_type = tk.StringVar(value="Line")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = ttk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=False, padx=10)

right_frame = ttk.Frame(main_frame)
right_frame.pack(side="right", fill="both", expand=True)

try:
    logo_image = Image.open("foxorox.png")
    logo_image = logo_image.resize((120, 120), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(left_frame, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack(pady=10)
except Exception as e:
    print(f"Failed to load icon: {e}")

ttk.Label(left_frame, text="📊 Select S&P 500 company:").pack()
ttk.Combobox(left_frame, textvariable=company_var, values=list(sp500_top10.keys()), state="readonly").pack()

ttk.Label(left_frame, text="📉 Gap threshold (%)").pack()
ttk.Entry(left_frame, textvariable=threshold).pack()

ttk.Label(left_frame, text="🧠 CatBoost iterations").pack()
ttk.Entry(left_frame, textvariable=iterations).pack()

ttk.Label(left_frame, text="🧠 Model depth").pack()
ttk.Entry(left_frame, textvariable=depth).pack()

ttk.Label(left_frame, text="🚀 Learning Rate").pack()
ttk.Entry(left_frame, textvariable=learning_rate).pack()

ttk.Button(left_frame, text="🔍 Analyze", command=update_prediction).pack(pady=5)
ttk.Button(left_frame, text="🗂 Calculate all", command=calculate_all).pack(pady=5)

ttk.Label(left_frame, textvariable=result_var, foreground="blue").pack(pady=10)

# --- Tabela wyników (treeview) ---
columns = ("Company", "Gap Probability", "Candle Color", "Candle Probability")

tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=10)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120, anchor='center')
tree.pack(pady=10, fill='x')

scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side='right', fill='y')

ttk.Label(left_frame, text="Chart type:").pack()
ttk.Radiobutton(left_frame, text="Line", variable=plot_type, value="Line", command=on_plot_type_change).pack(anchor="w")
ttk.Radiobutton(left_frame, text="Candlestick", variable=plot_type, value="Candlestick", command=on_plot_type_change).pack(anchor="w")

chart_frame = ttk.Frame(right_frame)
chart_frame.pack(fill="both", expand=True)

# Start with initial calculation
update_prediction()

root.mainloop()
