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

# Funkcja normalizująca prawdopodobieństwa
def normalize_probability(prob):
    if prob < 0.4:
        return 0  # Brak pewności
    elif prob > 0.6:
        return 1  # Pełna pewność
    else:
        # Skala od 0 do 1 dla przedziału 40%-60%
        return (prob - 0.4) / (0.6 - 0.4)

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
    results.append(next_probs)
    print(f"Przetworzono dane dla: {name} ({ticker})")

# Zapisz scalone wyniki
final_df = pd.concat(results, ignore_index=True)
final_df = final_df.melt(id_vars=['Company', 'Ticker'], var_name='Next_State', value_name='Probability')
final_df = final_df.pivot_table(index=['Company', 'Ticker'], columns='Next_State', values='Probability')
final_df = final_df.reset_index()

# === RYSUJ WYKRES SŁUPKOWY ===
plt.figure(figsize=(20, 15))

# Przetwarzamy dane dla wykresu
for i, row in final_df.iterrows():
    # Wzrosty
    if 1 in row and row[1] > 0:  # Tylko gdy prawdopodobieństwo wzrostu > 0
        prob = row[1]
        normalized_prob = normalize_probability(prob)
        plt.bar(row['Company'], normalized_prob, color='green', label=row['Company'], bottom=0)
    # Spadki
    if -1 in row and row[-1] > 0:  # Tylko gdy prawdopodobieństwo spadku > 0
        prob = row[-1]
        normalized_prob = normalize_probability(prob)
        plt.bar(row['Company'], -normalized_prob, color='red', label=row['Company'], bottom=0)

# Dodajemy tytuł, legendę i etykiety
plt.title("Prawdopodobieństwa wzrostów i spadków dla spółek WIG20", fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\nWyniki zapisane w wig20_probabilities.csv")
