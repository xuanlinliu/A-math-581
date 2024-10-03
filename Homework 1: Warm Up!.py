import numpy as np
def f(x):
    return x * np.sin(3*x) - np.exp(x)
def df(x):
    return np.sin(3*x) + 3*x*np.cos(3*x)-np.exp(x)

def newton_raphson(f, df, x0, tol=1e-6, max_iter=1000):
    x_values = [x0]
    for i in range(max_iter):
        x1 = x0 - (f(x0)/df(x0))
        A1 = x_values.append(x1)
        if abs(f(x1)) < tol:
            break
        x0 = x1
    return x_values, i+1

def bisection(f, a, b, tol=1e-6, max_iter=1000):
    mid_points = []
    for j in range(max_iter):
        mid = (a+b)/2
        A2 = mid_points.append(mid)
        if abs(f(mid)) < tol or (b-a)/2 < tol:
            break
        if np.sign(f(mid)) == np.sign(f(a)):
            a = mid
        else:
            b = mid
    return mid_points, j
x0 = -1.6
newton_values, newton_iter = newton_raphson(f, df, x0)
a = -0.7
b = -0.4
bisection_values, bisection_iter = bisection(f, a, b)
np.save('A1.npy', newton_values)
np.save('A2.npy', bisection_values)
np.save('A3.npy', [newton_iter, bisection_iter])
A3 = [newton_iter, bisection_iter]
print("Newton-Raphson iterations", newton_iter)
print("Bisection iterations:", bisection_iter)

# Part 2 - Matrix Operations
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

# (a) A + B
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
A4 = A + B
print(A4)
np.save('A4.npy', A4)

# (b) 3x - 4y
x = np.array([[1], [0]])
y = np.array([[0], [1]])
A5 = 3 * x - 4 * y
print(A5)
np.save('A5.npy', A5)

# (c) Ax
A = np.array([[1, 2], [-1, 1]])
x = np.array([[1], [0]])
A6 = A @ x
print(A6)
np.save('A6.npy', A6)

# (d) B(x - y)
B = np.array([[2, 0], [0, 2]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
A7 = B @ (x - y)
print(A7)
np.save('A7.npy', A7)

# (e) Dx
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
A8 = D @ x
print(A8)
np.save('A8.npy', A8)

# (f) Dy + z
D = np.array([[1, 2], [2, 3], [-1, 0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])
A9 = D @ y + z
print(A9)
np.save('A9.npy', A9)

# (g) AB
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
A10 = A @ B
print(A10)
np.save('A10.npy', A10)

# (h) BC
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
A11 = B @ C
np.save('A11.npy', A11)

# (i) CD
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
A12 = C @ D
print(A12)
np.save('A12.npy', A12)


print("A4.npy (A + B):", np.load('A4.npy'))
print("A5.npy (3x - 4y):", np.load('A5.npy'))
print("A6.npy (Ax):", np.load('A6.npy'))
print("A7.npy (B(x - y)):", np.load('A7.npy'))
print("A8.npy (Dx):", np.load('A8.npy'))
print("A9.npy (Dy + z):", np.load('A9.npy'))
print("A10.npy (AB):", np.load('A10.npy'))
print("A11.npy (BC):", np.load('A11.npy'))
print("A12.npy (CD):", np.load('A12.npy'))