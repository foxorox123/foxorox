# Funkcja kalkulatora
def calculator():
    print("Wybierz operację:")
    print("1. Dodawanie")
    print("2. Odejmowanie")
    print("3. Mnożenie")
    print("4. Dzielenie")
    
    # Wybór operacji
    choice = input("Wpisz numer operacji (1/2/3/4): ")

    # Wprowadź liczby
    num1 = float(input("Wprowadź pierwszą liczbę: "))
    num2 = float(input("Wprowadź drugą liczbę: "))

    # Operacje
    if choice == '1':
        print(f"Wynik: {num1} + {num2} = {num1 + num2}")
    elif choice == '2':
        print(f"Wynik: {num1} - {num2} = {num1 - num2}")
    elif choice == '3':
        print(f"Wynik: {num1} * {num2} = {num1 * num2}")
    elif choice == '4':
        if num2 != 0:
            print(f"Wynik: {num1} / {num2} = {num1 / num2}")
        else:
            print("Błąd: Dzielenie przez zero!")
    else:
        print("Nieprawidłowy wybór operacji.")

# Uruchomienie kalkulatora
calculator()

# Dodanie pauzy, aby użytkownik mógł zobaczyć wynik przed zamknięciem
input("Naciśnij dowolny klawisz, aby zamknąć...")