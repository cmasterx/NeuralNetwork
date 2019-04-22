import cmath
import numpy as np
import matplotlib.pyplot as plt


# k - k value
# T0 - period
# t - time
def ak(coe, k, T0, t):
    return coe * cmath.exp(1j * 2 * cmath.pi / T0 * k * t)


def compute_fourier_series(coe, k, T0, t):
    return [sum(ak(coe[j], k[j], T0, t[i]) for j in range(len(coe))) for i in range(len(t))]


def compute_xna(coe, T0, t, min_k=-1000, max_k=1000):
    compute_xna(coe, T0, t, np.arange(min_k, max_k + 1))


def compute_xna(coe, T0, t, k_list):
    # k_list = np.arange(min_k, max_k + 1)

    total = np.array(compute_fourier_series(coe, [k_list[0]] * len(coe), T0, t))

    for k in k_list[1:]:
        total = total + np.array(compute_fourier_series(coe, [k] * len(coe), T0, t))

    return total


x = np.arange(0, 100, 0.1)
y = compute_fourier_series([0.5, 0.5, 5, -5], [2, -2, -2j, 2j], 6, x)

# imgList = []
# for i in y:
#     if i.imag != 0:
#         imgList.append(i.imag)
#
# print(imgList)

# test
T0 = 3
N = [3, 5, 11, 32, 100]
t = np.arange(-3, 3, 0.01)

xna = []
for i in N:
    xna.append(compute_fourier_series([1], [i], T0, t))


# xn = [ sum(j[t] for j in xna) for i in range(len(t))]

# # square wave test
# w0 = cmath.pi
# min_lim = 1
# max_lim = 100
# t_sq = np.arange(1.5, 5, 0.01)
# k_sq = np.arange(min_lim, max_lim + 1, 1)
# coef = [ ( (-1)/(2j * _k * w0)*(1 - cmath.exp(-1j * _k * w0)) ) for _k in k_sq]
# sq_wav = compute_fourier_series(coef, k_sq, 2, t_sq)
# plt.plot(t_sq, sq_wav)
#
# yn1 = compute_xna(coef, T0, t_sq, -500, 500)
# yn1 = [i.real for i in yn1]
# print(len(t_sq))
# plt.plot(t_sq, yn1)
#
# # plt.plot(t, compute_fourier_series([1], [3], T0, t))
# # plt.plot(t, xn)
# # plt.plot(t, compute_fourier_series([1], [5], T0, t))
# # plt.plot(t, compute_fourier_series([1], [11], T0, t))

def problem_coef(k, w0):
    if k != 0:
        return (1 / (3j * k * w0)) * (cmath.exp(1j * k * w0 * 0.75) - cmath.exp(-1j * k * w0 * 0.75))
    else:
        return 1 # temporary

k1 = []
T1 = 3
W1 = np.pi * 2 / T1
t1 = np.arange(-3, 3, 0.01)
N = [3, 5, 11, 32, 100]
coe1 = [problem_coef(i, W1) for i in N]
yres = compute_fourier_series(coe1, N, T1, t1)

for k in N:
    k_range = np.arange(-k, k + 1)
    coeficients = [problem_coef(c, W1) for c in k_range]

    yres = compute_xna(coeficients, W1, t1, k_range)
    plt.plot(t1, yres)

# yres = compute_fourier_series([problem_coef(3, W1)], [3], T1, t1)

plt.plot(t1, yres)

plt.show()
