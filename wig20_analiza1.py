import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === KONFIGURACJA ===
DATA_DIR = r"C:\Users\aerga\Desktop\dane\wse stocks"
OUTPUT_DIR = r"C:\Users\aerga\Desktop\dane\wyniki"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapowanie spółek WIG20 do kodów GPW
WIG20_TICKERS = {
    "alior": "alr",
    "allegro": "ale",
    "budimex": "bdx",
    "ccc": "ccc",
    "cd projekt": "cdr",
    "dino": "dnp",
    "kghm": "kgh",
    "kruk": "kru",
    "lpp": "lpp",
    "mbank": "mbk",
    "orange": "opl",
    "pekao": "pko",
    "pepco": "pep",
    "pge": "pge",
    "pkn": "pkn",
    "pko": "pko",
    "pzu": "pzu",
    "santander": "spl"
}

# === PRZETWARZANIE DANYCH ===
results = []

for name, ticker in WIG20_TICKERS.items():
    data_file = os.path.join(DATA_DIR, f"{ticker}.txt")
    if not os.path.isfile(data_file):
        print(f"Brak danych dla spółki: {name} ({ticker})")
        continue
    try:
        df = pd.read_csv(data_file, sep=',', header=0)
    except PermissionError as e:
        print(f"Brak dostępu do pliku: {data_file}")
        continue
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku {data_file}: {e}")
        continue
    
    df['datetime'] = pd.to_datetime(df['<DATE>'].astype(str) + df['<TIME>'].astype(str), format='%Y%m%d%H%M%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    df['price_change'] = df['<CLOSE>'].diff()
    df['state'] = df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Obliczanie SMA5 i SMA21
    df['SMA5'] = df['<CLOSE>'].rolling(window=5).mean()
    df['SMA21'] = df['<CLOSE>'].rolling(window=21).mean()

    # Sprawdź, czy są wystarczające dane do obliczenia crossoverów
    df['Crossovers'] = np.nan
    df['Crossovers'] = np.where(df['SMA5'] > df['SMA21'], 1, df['Crossovers'])
    df['Crossovers'] = np.where(df['SMA5'] < df['SMA21'], -1, df['Crossovers'])
    
    # Upewnijmy się, że kolumna 'Crossovers' jest obecna
    if 'Crossovers' not in df.columns:
        df['Crossovers'] = 0  # Przypisanie wartości 0, jeśli nie ma crossoveru
    
    # Przypisanie ostatniego crossoveru dla całej spółki
    last_crossover = df['Crossovers'].dropna().iloc[-1] if not df['Crossovers'].dropna().empty else 0
    df['Crossovers'] = last_crossover  # Stosujemy ostatni crossover w całym zbiorze

    states = [-1, 0, 1]
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    for i in range(1, len(df)):
        prev_state = df.loc[i - 1, 'state']
        curr_state = df.loc[i, 'state']
        transition_matrix.loc[prev_state, curr_state] += 1

    transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    next_probs = transition_probs.reset_index()
    next_probs['Company'] = name
    next_probs['Ticker'] = ticker
    next_probs['Crossovers'] = last_crossover  # Ostatni crossover dla spółki

    results.append(next_probs)
    print(f"Przetworzono dane dla: {name} ({ticker})")

# Zapisz scalone wyniki
final_df = pd.concat(results, ignore_index=True)
final_df = final_df.melt(id_vars=['Company', 'Ticker', 'index', 'Crossovers'], var_name='Next_State', value_name='Probability')
final_df = final_df.pivot_table(index=['Company', 'Ticker'], columns='Next_State', values='Probability')
final_df = final_df.reset_index()

# === RYSUJ WYKRES SŁUPKOWY ===
plt.figure(figsize=(20, 15))

for i, row in final_df.iterrows():
    states = []
    probs = []
    colors = []
    crossovers = row['Crossovers']
    
    for state in [-1, 1]:
        prob = row[state] if state in row else 0
        if prob > 0:  # Omiń stany zerowe
            states.append(state)
            probs.append(prob)
            colors.append('green' if state == 1 else 'red')
    
    # Wykresy słupkowe
    plt.bar([row['Company']] * len(states), probs, color=colors, label=row['Company'])

    # Dodaj trójkąt na wykresie, jeżeli doszło do crossoveru
    if crossovers == 1:
        plt.plot([row['Company']], [max(probs)], marker='^', color='blue', markersize=10, label="SMA5/SMA21 crossover")
    elif crossovers == -1:
        plt.plot([row['Company']], [min(probs)], marker='v', color='blue', markersize=10, label="SMA5/SMA21 crossover")

plt.title("Prawdopodobieństwa wzrostów i spadków dla spółek WIG20 oraz crossover SMA5/SMA21", fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Zapisz dane do pliku CSV
final_df.to_csv(os.path.join(OUTPUT_DIR, "wig20_probabilities_with_crossovers.csv"), index=False)

print("\nWyniki zapisane w wig20_probabilities_with_crossovers.csv")
