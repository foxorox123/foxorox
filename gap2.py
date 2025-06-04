import pandas as pd
import tkinter as tk
from tkinter import ttk
from catboost import CatBoostClassifier
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk

# === Mapa nazw spółek WIG20 → symboli na stooq.pl ===
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
        print(f"❌ Błąd pobierania danych dla {symbol}: {e}")
        return pd.DataFrame()

def train_gap_model(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['GapAboveThreshold']
    if y.nunique() <= 1:
        raise ValueError("Target dla gap_model zawiera tylko jedną unikalną wartość.")
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
    if y.nunique() <= 1:
        raise ValueError("Target dla candle_model zawiera tylko jedną unikalną wartość.")
    model = CatBoostClassifier(
        iterations=iterations.get(),
        depth=depth.get(),
        learning_rate=learning_rate.get(),
        verbose=False
    )
    model.fit(X, y)
    return model

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
            f"📊 {company_var.get()}:\n"
            f" - Szansa na lukę > {threshold.get()}%: {gap_proba*100:.2f}%\n"
            f" - Przewidywana następna świeca: {candle_color}\n"
            f" - Prawdopodobieństwo: {candle_proba*100:.2f}%"
        )
    except Exception as e:
        result_var.set("❌ Błąd analizy.\nSzczegóły w 'errors.log'.")
        with open("errors.log", "a") as f:
            f.write(f"[{datetime.now()}] {type(e).__name__}: {str(e)}\n")

def calculate_all():
    for row in tree.get_children():
        tree.delete(row)

    results = []

    for company, symbol in wig20_symbols.items():
        df = fetch_data(symbol)
        if df.empty:
            results.append((company, None, None, None))
            continue
        try:
            gap_model = train_gap_model(df)
            candle_model = train_candle_model(df)
            latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]

            gap_proba = gap_model.predict_proba(latest)[0][1]
            candle_proba = candle_model.predict_proba(latest)[0][1]
            candle_color = "biała" if candle_proba > 0.5 else "czarna"

            results.append((company, gap_proba*100, candle_color, candle_proba*100))
        except Exception as e:
            print(f"⚠️ Pominięto {company}: {e}")
            results.append((company, None, None, None))

    results.sort(key=lambda x: (x[3] is None, -x[3] if x[3] is not None else 0))

    for company, gap_pct, candle_col, candle_pct in results:
        if gap_pct is None:
            tree.insert("", "end", values=(company, "Brak danych", "-", "-"))
        else:
            tree.insert("", "end", values=(
                company,
                f"{gap_pct:.2f}%",
                candle_col,
                f"{candle_pct:.2f}%"
            ))

# === GUI ===
root = tk.Tk()
root.title("📈 WIG20 Gap i świeca Predictor")
root.geometry("600x700")

# === Logo ===
try:
    logo_path = "C:/Users/aerga/Desktop/dane/foxoro.png"
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((100, 100), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(root, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack(pady=5)
except Exception as e:
    print(f"❌ Błąd ładowania logo: {e}")

# === Zmienne GUI ===
threshold = tk.DoubleVar(value=1.0)
iterations = tk.IntVar(value=100)
depth = tk.IntVar(value=6)
learning_rate = tk.DoubleVar(value=0.1)
company_var = tk.StringVar(value=list(wig20_symbols.keys())[0])
result_var = tk.StringVar()

# === GUI Layout ===
ttk.Label(root, text="📊 Wybierz spółkę z WIG20:").pack()
ttk.Combobox(root, textvariable=company_var, values=list(wig20_symbols.keys()), state="readonly").pack()

ttk.Label(root, text="📉 Próg gapu (%)").pack()
ttk.Entry(root, textvariable=threshold).pack()

ttk.Label(root, text="🧠 Iteracje CatBoost").pack()
ttk.Entry(root, textvariable=iterations).pack()

ttk.Label(root, text="🧠 Głębokość modelu").pack()
ttk.Entry(root, textvariable=depth).pack()

ttk.Label(root, text="🚀 Learning Rate").pack()
ttk.Entry(root, textvariable=learning_rate).pack()

ttk.Button(root, text="🔍 Analizuj", command=update_prediction).pack(pady=5)
ttk.Button(root, text="🗂 Przelicz dla wszystkich", command=calculate_all).pack(pady=5)

ttk.Label(root, textvariable=result_var, foreground="blue", font=("Arial", 12), wraplength=580).pack(pady=10)

columns = ("Spółka", "Szansa na gap > próg", "Następna świeca", "Prawdopodobieństwo świecy (%)")
tree = ttk.Treeview(root, columns=columns, show='headings', height=15)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=140, anchor=tk.CENTER)
tree.pack(padx=10, pady=10, fill='both', expand=True)

root.mainloop()
