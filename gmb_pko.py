import pandas as pd

# 1. Wczytaj dane z pliku CSV
df = pd.read_csv("pko.csv", sep=';')

# 2. Stwórz kolumnę datetime
df['datetime'] = pd.to_datetime(df['<DATE>'].astype(str) + df['<TIME>'].astype(str), format='%Y%m%d%H%M%S')
df = df.sort_values('datetime').reset_index(drop=True)

# 3. Oblicz zmiany cen i przypisz stany
df['price_change'] = df['<CLOSE>'].diff()
df['state'] = df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# 4. Zbuduj macierz przejść
states = [-1, 0, 1]
transition_matrix = pd.DataFrame(0, index=states, columns=states)

for i in range(1, len(df)):
    prev_state = df.loc[i - 1, 'state']
    curr_state = df.loc[i, 'state']
    transition_matrix.loc[prev_state, curr_state] += 1

# 5. Przelicz na prawdopodobieństwa
transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

# 6. Przewidź następny stan na podstawie ostatniego
last_state = df['state'].iloc[-1]
next_state = transition_probs.loc[last_state].idxmax()
next_state_probabilities = transition_probs.loc[last_state]

print("Macierz przejść:")
print(transition_probs)
print(f"\nOstatni stan: {last_state}")
print(f"Najbardziej prawdopodobny następny stan: {next_state}")
print(f"Prawdopodobieństwa: \n{next_state_probabilities}")
