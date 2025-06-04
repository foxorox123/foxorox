import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score

# Załaduj dane z pliku CSV
data = pd.read_csv("ścieżka/do/pliku.csv")  # Ustaw właściwą ścieżkę do pliku z danymi

# Przypisz dane do X (cechy) i y (target)
# Załóżmy, że ostatnia kolumna to target, a pozostałe to cechy:
X = data.drop('target', axis=1)  # Zastąp 'target' nazwą kolumny z etykietami
y = data['target']  # Zastąp 'target' nazwą kolumny z etykietami

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Stwórz i wytrenuj model klasyfikatora
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcje
y_pred = model.predict(X_test)

# Ocena modelu
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# F1 Score (macro average)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"F1 Macro: {f1_macro:.4f}")

# ROC AUC Score (dla wieloklasowej klasyfikacji)
try:
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc:.4f}")
except ValueError:
    print("ROC AUC Score could not be computed.")
