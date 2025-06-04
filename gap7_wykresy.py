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

wig20_symbols = {
    'ALLEGRO': 'ale',
    'ALIOR': 'alr',
    'CCC': 'ccc',
    'CD PROJEKT': 'cdr',
    'CYFROWY POLSAT': 'cps',
    'DINO': 'dnp',
    'JSW': 'jsw',
    'KGHM': 'kgh',
    'KRUK': 'kru',
    'LPP': 'lpp',
    'MBANK': 'mbk',
    'ORLEN': 'pkn',
    'PKO BP': 'pko',
    'PZU': 'pzu',
    'SANPL': 'spl',
    'SANTANDER': 'san'
}

def fetch_data(symbol):
    try:
        url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
        data = pd.read_csv(url, parse_dates=['Data'], encoding='ISO-8859-2')
        data = data.rename(columns={
            'Data': 'Date',
            'Otwarcie': 'Open',
            'Najwyzszy': 'High',
            'Najnizszy': 'Low',
            'Zamkniecie': 'Close',
            'Wolumen': 'Volume'
        })
        data = data.sort_values("Date").dropna()

        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        data['GapAboveThreshold'] = (data['Gap'].abs() > threshold.get() / 100).astype(int)
        data['NextCandleWhite'] = (data['Close'].shift(-1) > data['Open'].shift(-1)).astype(int)

        return data.dropna().tail(200)
    except Exception as e:
        print(f"Błąd pobierania danych dla {symbol}: {e}")
        return pd.DataFrame()

def train_gap_model(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['GapAboveThreshold']
    if len(y.unique()) == 1:
        raise ValueError("Tylko jedna unikalna wartość w target")
    model = CatBoostClassifier(
        iterations=iterations.get(),
        depth=depth.get(),
        learning_rate=learning_rate.get(),
        verbose=False
    )
    model.fit(X, y)
    return model

def train_candle_model(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['NextCandleWhite']
    if len(y.unique()) == 1:
        raise ValueError("Tylko jedna unikalna wartość w target")
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

# 🔁 Zmieniona funkcja show_plot
def show_plot(df, title):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(7, 6), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.05, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    data_tail = df.tail(50).copy()
    data_tail['Volume'] = data_tail['Volume'] / 1_000_000  # dzielimy wolumen

    if plot_type.get() == "Liniowy":
        ax1.plot(data_tail['Date'], data_tail['Close'], label="Zamknięcie", color='blue')
    else:
        plot_candlestick(ax1, data_tail)

    ax1.set_title(title)
    ax1.set_ylabel("Cena")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    ax2.bar(data_tail['Date'], data_tail['Volume'], color='gray', width=0.6)
    ax2.set_ylabel("Wolumen (mln PLN)")
    ax2.grid(True)

    fig.autofmt_xdate(rotation=45)
    fig.subplots_adjust(hspace=0.1)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def update_prediction():
    try:
        df = fetch_data(wig20_symbols[company_var.get()])
        if df.empty:
            result_var.set("Brak danych")
            return

        gap_model = train_gap_model(df)
        candle_model = train_candle_model(df)
        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]

        gap_proba = gap_model.predict_proba(latest)[0][1]
        candle_proba = candle_model.predict_proba(latest)[0][1]
        candle_color = "biała (wzrostowa)" if candle_proba > 0.5 else "czarna (spadkowa)"

        result_var.set(
            f"{company_var.get()}:\n"
            f" - Szansa na lukę > {threshold.get()}%: {gap_proba*100:.2f}%\n"
            f" - Przewidywana następna świeca: {candle_color}\n"
            f" - Prawdopodobieństwo: {candle_proba*100:.2f}%"
        )
        show_plot(df, f"Wykres: {company_var.get()}")
    except Exception as e:
        result_var.set("❌ Błąd analizy.")
        with open("errors.log", "a") as f:
            f.write(f"[{datetime.now()}] {type(e).__name__}: {str(e)}\n")

def calculate_all():
    for row in tree.get_children():
        tree.delete(row)

    results = []

    for company, symbol in wig20_symbols.items():
        df = fetch_data(symbol)
        if df.empty:
            results.append((company, None, None, None, df))
            continue
        try:
            gap_model = train_gap_model(df)
            candle_model = train_candle_model(df)
        except:
            results.append((company, None, None, None, df))
            continue

        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]
        gap_proba = gap_model.predict_proba(latest)[0][1]
        candle_proba = candle_model.predict_proba(latest)[0][1]
        candle_color = "biała" if candle_proba > 0.5 else "czarna"

        results.append((company, gap_proba*100, candle_color, candle_proba*100, df))

    results.sort(key=lambda x: (x[3] is None, -x[3] if x[3] is not None else 0))

    for company, gap_pct, candle_col, candle_pct, _ in results:
        if gap_pct is None:
            tree.insert("", "end", values=(company, "Brak danych", "-", "-"))
        else:
            tree.insert("", "end", values=(company, f"{gap_pct:.2f}%", candle_col, f"{candle_pct:.2f}%"))

    for company, gap_pct, candle_col, candle_pct, df in results:
        if candle_pct is not None:
            show_plot(df, f"Wykres: {company} (najwyższe prawd.)")
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
company_var = tk.StringVar(value=list(wig20_symbols.keys())[0])
result_var = tk.StringVar()
plot_type = tk.StringVar(value="Liniowy")

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
    print(f"Nie udało się wczytać ikony: {e}")

ttk.Label(left_frame, text="📊 Wybierz spółkę z WIG20:").pack()
ttk.Combobox(left_frame, textvariable=company_var, values=list(wig20_symbols.keys()), state="readonly").pack()

ttk.Label(left_frame, text="📉 Próg gapu (%)").pack()
ttk.Entry(left_frame, textvariable=threshold).pack()

ttk.Label(left_frame, text="🧠 Iteracje CatBoost").pack()
ttk.Entry(left_frame, textvariable=iterations).pack()

ttk.Label(left_frame, text="🧠 Głębokość modelu").pack()
ttk.Entry(left_frame, textvariable=depth).pack()

ttk.Label(left_frame, text="🚀 Learning Rate").pack()
ttk.Entry(left_frame, textvariable=learning_rate).pack()

ttk.Button(left_frame, text="🔍 Analizuj", command=update_prediction).pack(pady=5)
ttk.Button(left_frame, text="🗂 Przelicz dla wszystkich", command=calculate_all).pack(pady=5)

ttk.Label(left_frame, textvariable=result_var, foreground="blue", font=("Arial", 12), wraplength=300).pack(pady=10)

columns = ("Spółka", "Szansa na gap > próg", "Następna świeca", "Prawdopodobieństwo świecy (%)")
tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=15)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=140, anchor=tk.CENTER)
tree.pack(padx=5, pady=5)

plot_type_frame = ttk.Frame(right_frame)
plot_type_frame.pack(fill='x', padx=10, pady=(10,0), anchor='ne')

ttk.Label(plot_type_frame, text="Typ wykresu: ").pack(side='left')
ttk.Radiobutton(plot_type_frame, text="Liniowy", variable=plot_type, value="Liniowy", command=on_plot_type_change).pack(side='left', padx=5)
ttk.Radiobutton(plot_type_frame, text="Świecowy", variable=plot_type, value="Świecowy", command=on_plot_type_change).pack(side='left', padx=5)

chart_frame = ttk.Frame(right_frame)
chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

root.mainloop()
