import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Wczytaj dane i przygotuj ===
df = pd.read_csv("pko.csv", sep=';')
df['datetime'] = pd.to_datetime(df['<DATE>'].astype(str) + df['<TIME>'].astype(str), format='%Y%m%d%H%M%S')
df = df.sort_values('datetime').reset_index(drop=True)

# === 2. Model MARKOWA ===
df['price_change'] = df['<CLOSE>'].diff()
df['state'] = df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

states = [-1, 0, 1]
transition_matrix = pd.DataFrame(0, index=states, columns=states)

for i in range(1, len(df)):
    prev_state = df.loc[i - 1, 'state']
    curr_state = df.loc[i, 'state']
    transition_matrix.loc[prev_state, curr_state] += 1

transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
last_state = df['state'].iloc[-1]
next_state = transition_probs.loc[last_state].idxmax()
next_probs = transition_probs.loc[last_state]

print("Macierz przejść:")
print(transition_probs)
print(f"\nOstatni stan: {last_state}")
print(f"Najbardziej prawdopodobny następny stan: {next_state}")
print(f"Prawdopodobieństwa:\n{next_probs}\n")

# === 3. Model GBM (Geometric Brownian Motion) ===
prices = df['<CLOSE>'].values
log_returns = np.diff(np.log(prices))

mu = np.mean(log_returns)
sigma = np.std(log_returns)
S0 = prices[-1]  # ostatnia cena

T = 6   # liczba godzin do przodu
n_simulations = 100
dt = 1

np.random.seed(42)
simulations = []

for _ in range(n_simulations):
    shocks = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=T)
    path = S0 * np.exp(np.cumsum(shocks))
    simulations.append(path)

# === 4. Wykres symulacji GBM ===
plt.figure(figsize=(10, 5))
for path in simulations:
    plt.plot(path, color='blue', alpha=0.1)

plt.title("Symulacje Geometric Brownian Motion – PKO BP")
plt.xlabel("Godziny w przyszłości")
plt.ylabel("Cena akcji")
plt.grid(True)
plt.show()
