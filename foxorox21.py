import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
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
import pytz
import os



# Pobierz pełną listę spółek S&P 500 ze strony Wikipedii
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    sp500_all = dict(zip(df['Security'], df['Symbol']))
    return sp500_all

sp500_top10 = get_sp500_tickers()

# --- Global storage for fetched data to avoid repeated downloads ---
company_data_cache = {}

def fix_symbol_yahoo(symbol):
    # Zamień '.' na '-' (np. BRK.B -> BRK-B) dla symboli z kropką
    if '.' in symbol:
        return symbol.replace('.', '-')
    return symbol

from datetime import datetime, time as dtime

def is_market_open():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    eastern = pytz.timezone('US/Eastern')
    now_est = now_utc.astimezone(eastern)
    market_open = dtime(9, 30)
    market_close = dtime(16, 0)
    return market_open <= now_est.time() <= market_close

DATA_DIR = "SP500_data"
MAX_DAYS_OLD = 10

# Tworzymy katalog jeśli nie istnieje
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_data(symbol):
    symbol_yahoo = fix_symbol_yahoo(symbol)
    file_path = os.path.join(DATA_DIR, f"{symbol_yahoo}.csv")

    def download_full():
        print(f"📥 Full download for {symbol_yahoo}")
        df = yf.Ticker(symbol_yahoo).history(period="300d")
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(file_path, index=False)
        return df

    # Jeśli plik istnieje — próbujemy go wczytać
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        last_date = df['Date'].max().date()
        today = datetime.today().date()
        days_diff = (today - last_date).days

        if days_diff > MAX_DAYS_OLD:
            return download_full()

        # Jeśli dzisiejsze dane już są – zwróć
        if last_date >= today:
            return df

        # ⛔ Jeśli giełda jest zamknięta, a dzisiejszych danych brak — nie pobieraj
        if not is_market_open():
            print(f"⛔ Market closed. Skipping update for {symbol_yahoo}")
            return df

        # Jeśli nie ma dzisiejszych danych – dograj brak
        print(f"🔄 Updating {symbol_yahoo} for today")
        new_data = yf.Ticker(symbol_yahoo).history(start=last_date + pd.Timedelta(days=1))
        if not new_data.empty:
            new_data.reset_index(inplace=True)
            new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = pd.concat([df, new_data], ignore_index=True)
            df.drop_duplicates('Date', keep='last', inplace=True)
            df.sort_values('Date', inplace=True)
            df.to_csv(file_path, index=False)

        return df

    return download_full()

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

def plot_bar_chart(ax, data):
    width = 0.4
    for idx, row in data.iterrows():
        date_num = mdates.date2num(row['Date'])
        open_, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        ax.plot([date_num, date_num], [low, high], color='black', linewidth=1)
        ax.plot([date_num - width, date_num], [open_, open_], color='green', linewidth=3)
        ax.plot([date_num, date_num + width], [close, close], color='red', linewidth=3)

def show_plot(df, title):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(7, 6), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    data_tail = df.tail(50).copy()
    data_tail['Volume'] = data_tail['Volume'] / 1_000_000

    if plot_type.get() == "Line":
        ax1.plot(data_tail['Date'], data_tail['Close'], label="Close", color='blue')
    elif plot_type.get() == "Candlestick":
        plot_candlestick(ax1, data_tail)
    elif plot_type.get() == "Bar":
        plot_bar_chart(ax1, data_tail)

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

    # Po wyliczeniu pokaż wykres spółki z najwyższym prawdopodobieństwem świecy
    for company, gap_pct, candle_col, candle_pct, df in results:
        if candle_pct is not None:
            show_plot(df, f"Chart: {company} (highest probability)")
            break

def on_plot_type_change():
    # Po zmianie typu wykresu odśwież wykres aktualnej spółki (z pola tekstowego result_var)
    # Możemy spróbować odczytać nazwę spółki z result_var
    try:
        text = result_var.get()
        if ':' in text:
            company = text.split(':')[0]
            symbol = sp500_top10.get(company, None)
            if symbol:
                df = fetch_data(symbol)
                if not df.empty:
                    show_plot(df, f"Chart: {company}")
    except:
        pass

def sort_treeview(tree, col, reverse, is_percentage=False):
    for c in tree["columns"]:
        tree.heading(c, text=c, command=lambda _c=c: sort_treeview(tree, _c, False, is_percentage=(_c in ["Gap %", "Candle %"])))

    data = [(tree.set(k, col), k) for k in tree.get_children()]
    if is_percentage:
        def parse(x): 
            try:
                return float(x[0].replace('%', '').replace('No data', '-1')) if x[0] != "-" else -1
            except:
                return -1
        data.sort(key=parse, reverse=reverse)
    else:
        data.sort(key=lambda x: x[0], reverse=reverse)

    for index, (_, k) in enumerate(data):
        tree.move(k, '', index)

    arrow = " ▲" if not reverse else " ▼"
    tree.heading(col, text=col + arrow, command=lambda: sort_treeview(tree, col, not reverse, is_percentage))

def on_tree_select(event):
    selected = tree.selection()
    if not selected:
        return
    item = tree.item(selected[0])
    company = item['values'][0]
    symbol = sp500_top10.get(company)
    if symbol:
        df = fetch_data(symbol)
        if not df.empty:
            show_plot(df, f"Chart: {company}")

def on_tree_motion(event):
    region = tree.identify("region", event.x, event.y)
    if region == "cell" or region == "tree":
        row_id = tree.identify_row(event.y)
        if row_id:
            tree.selection_set(row_id)

# === GUI ===
root = tk.Tk()
# === MENU BAR ===
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Tools menu
tools_menu = tk.Menu(menu_bar, tearoff=0)
tools_menu.add_command(label="Calculate All", command=calculate_all)
tools_menu.add_command(label="Analyze Selected", command=update_prediction)
menu_bar.add_cascade(label="Tools", menu=tools_menu)

# Options menu
options_menu = tk.Menu(menu_bar, tearoff=0)
# Placeholder – można tu dodać np. ustawienia modelu
options_menu.add_command(label="Settings (TODO)", command=lambda: print("Settings menu clicked"))
menu_bar.add_cascade(label="Options", menu=options_menu)

# About menu
about_menu = tk.Menu(menu_bar, tearoff=0)
about_menu.add_command(label="About Foxorox", command=lambda: tk.messagebox.showinfo("About", "Foxorox AI Market Analyzer\nVersion 1.0"))
menu_bar.add_cascade(label="About", menu=about_menu)

root.title("Foxorox")
root.geometry("1200x700")

threshold = tk.DoubleVar(value=1.0)
iterations = tk.IntVar(value=100)
depth = tk.IntVar(value=6)
learning_rate = tk.DoubleVar(value=0.08)
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

ttk.Label(left_frame, text="🧠 AI iterations").pack()
ttk.Entry(left_frame, textvariable=iterations).pack()

ttk.Label(left_frame, text="🧠 AI Model depth").pack()
ttk.Entry(left_frame, textvariable=depth).pack()

ttk.Label(left_frame, text="🚀 AI Learning Rate").pack()
ttk.Entry(left_frame, textvariable=learning_rate).pack()

ttk.Button(left_frame, text="🔍 Analyze", command=update_prediction).pack(pady=5)
ttk.Button(left_frame, text="🗂 Calculate all", command=calculate_all).pack(pady=5)

ttk.Label(left_frame, textvariable=result_var, foreground="blue").pack(pady=10)

chart_control_frame = ttk.Frame(right_frame)
chart_control_frame.pack(fill='x', padx=10, pady=5)

ttk.Label(chart_control_frame, text="Chart type:").pack(side='left', padx=(0,10))
ttk.Radiobutton(chart_control_frame, text="Line", variable=plot_type, value="Line", command=on_plot_type_change).pack(side='left')
ttk.Radiobutton(chart_control_frame, text="Candlestick", variable=plot_type, value="Candlestick", command=on_plot_type_change).pack(side='left')
ttk.Radiobutton(chart_control_frame, text="Bar", variable=plot_type, value="Bar", command=on_plot_type_change).pack(side='left')

chart_frame = ttk.Frame(right_frame)
chart_frame.pack(fill="both", expand=True)

tree_frame = ttk.Frame(left_frame)
tree_frame.pack(pady=10, fill='both', expand=True)

tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
tree_scrollbar.pack(side='right', fill='y')

tree = ttk.Treeview(tree_frame, columns=("Company", "Gap %", "Candle Color", "Candle %"), show="headings", height=10, yscrollcommand=tree_scrollbar.set)
tree_scrollbar.config(command=tree.yview)

tree.heading("Company", text="Company", command=lambda: sort_treeview(tree, "Company", False))
tree.heading("Gap %", text="Gap %", command=lambda: sort_treeview(tree, "Gap %", False, is_percentage=True))
tree.heading("Candle Color", text="Candle Color", command=lambda: sort_treeview(tree, "Candle Color", False))
tree.heading("Candle %", text="Candle %", command=lambda: sort_treeview(tree, "Candle %", False, is_percentage=True))

tree.column("Company", width=150)
tree.column("Gap %", width=100)
tree.column("Candle Color", width=100)
tree.column("Candle %", width=100)

tree.pack(fill="both", expand=True)

# Event bind to update plot on selection in treeview
tree.bind("<<TreeviewSelect>>", on_tree_select)
# Optional: highlight row under mouse pointer
#tree.bind("<Motion>", on_tree_motion)

root.mainloop()