import pandas as pd
import tkinter as tk
from tkinter import ttk
from catboost import CatBoostClassifier
from datetime import datetime
import numpy as np

# === Mapa nazw spółek WIG20 → symboli na stooq.pl ===
wig20_symbols = {
    'ALLEGRO': 'ale',
    'ALIOR': 'alr',
    'CCC': 'ccc',
    'CD PROJEKT': 'cdr',
    'CYFROWY POLSAT': 'cps',
    'DINO': 'dnp',
    'GRUPA KĘTY': 'ket',
    'JSW': 'jsw',
    'KGHM': 'kgh',
    'KRUK': 'kru',
    'LOTOS': 'lto',
    'LPP': 'lpp',
    'MBANK': 'mbk',
    'ORLEN': 'pkn',
    'ORANGE': 'ope',
    'PEKAO': 'pek',
    'PKO BP': 'pko',
    'PZU': 'pzu',
    'SANPL': 'spl',
    'SANTANDER': 'san'
}

# === Pobieranie danych spółki ze stooq ===
def fetch_data():
    try:
        selected_name = company_var.get()
        symbol = wig20_symbols[selected_name].lower()
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
        return data.tail(200)
    except Exception as e:
        result_var.set(f"❌ Błąd pobierania danych:\n{e}")
        return pd.DataFrame()

# === Trening modelu CatBoost ===
def train_model(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['GapAboveThreshold']
    model = CatBoostClassifier(
        iterations=iterations.get(),
        depth=depth.get(),
        learning_rate=learning_rate.get(),
        verbose=False
    )
    model.fit(X, y)
    return model

# === Aktualizacja predykcji i wyświetlenie wyniku ===
def update_prediction():
    try:
        df = fetch_data()
        if df.empty:
            return
        model = train_model(df)
        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]
        proba = model.predict_proba(latest)[0][1]
        percent = round(proba * 100, 2)
        result_var.set(f"📊 {company_var.get()}: {percent}% szans,\n"
                       f"że kolejna sesja otworzy się z luką > {threshold.get()}%.")
    except Exception as e:
        result_var.set("❌ Błąd analizy.\nSzczegóły w 'errors.log'.")
        with open("errors.log", "a") as f:
            f.write(f"[{datetime.now()}] {type(e).__name__}: {str(e)}\n")

# === GUI ===
root = tk.Tk()
root.title("📈 WIG20 Gap Predictor")
root.geometry("520x360")

# Zmienne GUI
threshold = tk.DoubleVar(value=1.0)
iterations = tk.IntVar(value=100)
depth = tk.IntVar(value=6)
learning_rate = tk.DoubleVar(value=0.1)
company_var = tk.StringVar(value=list(wig20_symbols.keys())[0])
result_var = tk.StringVar()

# Widżety
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

ttk.Button(root, text="🔍 Analizuj", command=update_prediction).pack(pady=10)
ttk.Label(root, textvariable=result_var, foreground="blue", font=("Arial", 12), wraplength=500).pack(pady=10)

root.mainloop()
