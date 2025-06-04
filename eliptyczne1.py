# Parametry krzywej: y^2 = x^3 + 2x + 3 (mod 17)
p = 17
a = 2
b = 3

def points_on_curve():
    points = []
    for x in range(p):
        rhs = (x**3 + a*x + b) % p
        for y in range(p):
            if (y * y) % p == rhs:
                points.append((x, y))
    return points

# Wypisz punkty i zatrzymaj program
points = points_on_curve()
print("Punkty na krzywej eliptycznej:")
for pt in points:
    print(pt)

input("\nNaciśnij Enter, aby zakończyć...")
