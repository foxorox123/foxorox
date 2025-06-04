import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from catboost import CatBoostClassifier
from datetime import datetime
import numpy as np
import requests
import io
import threading

# Mapa nazw spółek WIG20 → symboli Stooq
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
    'SANTANDER': 'san'
}

def fetch_stock_data(symbol):
    url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'ISO-8859-2'
        if "Brak danych" in response.text or "404" in response.text or "<html" in response.text.lower():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(response.text), parse_dates=['Data'])
        df.rename(columns={
            'Data': 'Date',
            'Otwarcie': 'Open',
            'Najwyzszy': 'High',
            'Najnizszy': 'Low',
            'Zamkniecie': 'Close',
            'Wolumen': 'Volume'
        }, inplace=True)
        df.sort_values("Date", inplace=True)
        df.dropna(inplace=True)
        return df.tail(100)  # Ograniczamy do 100 dni
    except Exception as e:
        print(f"Błąd pobierania danych dla {symbol}: {e}")
        return pd.DataFrame()

def train_catboost(df, threshold_val):
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['GapAboveThreshold'] = (df['Gap'].abs() > threshold_val / 100).astype(int)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['GapAboveThreshold']
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False)
    model.fit(X, y)
    return model

def analyze_catboost_thread():
    threading.Thread(target=analyze_catboost).start()

def analyze_catboost():
    rows = []
    for name, symbol in wig20_symbols.items():
        df = fetch_stock_data(symbol)
        if df.empty or len(df) < 10:
            continue
        try:
            model = train_catboost(df, threshold.get())
            latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]
            proba = model.predict_proba(latest)[0][1]
            date = df['Date'].iloc[-1].strftime("%Y-%m-%d")
            open_price = df['Open'].iloc[-1]
            close_price = df['Close'].iloc[-1]
            rows.append([name, f"{proba*100:.2f}%", date, f"{open_price:.2f}", f"{close_price:.2f}"])
        except Exception as e:
            print(f"Błąd dla {name}: {e}")
    rows.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)
    root.after(0, lambda: update_table(rows, "CatBoost: Luka cenowa"))

def analyze_markov_thread():
    threading.Thread(target=analyze_markov).start()

def analyze_markov():
    rows = []
    for name, symbol in wig20_symbols.items():
        df = fetch_stock_data(symbol)
        if df.empty or len(df) < 20:
            continue
        df['Color'] = np.where(df['Close'] > df['Open'], 'White', 'Black')
        df['PrevColor'] = df['Color'].shift(1)
        markov = df.groupby('PrevColor')['Color'].value_counts(normalize=True).unstack()
        last_color = df['Color'].iloc[-1]
        if last_color in markov.index:
            prob = markov.loc[last_color].get('White', 0.0)
        else:
            prob = 0.5
        date = df['Date'].iloc[-1].strftime("%Y-%m-%d")
        open_price = df['Open'].iloc[-1]
        close_price = df['Close'].iloc[-1]
        rows.append([name, f"{prob*100:.2f}%", date, f"{open_price:.2f}", f"{close_price:.2f}"])
    rows.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)
    root.after(0, lambda: update_table(rows, "Markow: Świeca biała"))

def update_table(data, title):
    for row in tree.get_children():
        tree.delete(row)
    tree.heading("#0", text=title)
    for row in data:
        tree.insert("", tk.END, text=row[0], values=row[1:])

root = tk.Tk()
root.title("📈 WIG20 - Predykcja Gapów i Koloru Świec")
root.geometry("720x520")

threshold = tk.DoubleVar(value=1.0)

ttk.Label(root, text="📉 Próg gapu (%)").pack(pady=5)
ttk.Entry(root, textvariable=threshold).pack(pady=5)

ttk.Button(root, text="🔍 Analizuj (CatBoost)", command=analyze_catboost_thread).pack(pady=5)
ttk.Button(root, text="🔄 Analizuj (Markow)", command=analyze_markov_thread).pack(pady=5)

columns = ("Prawdopodobieństwo", "Data", "Cena Otwarcia", "Cena Zamknięcia")
tree = ttk.Treeview(root, columns=columns, show="tree headings", height=20)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=130)
tree.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()
