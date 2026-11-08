"""The two tests the report uses, standard library only: one-sided Fisher exact tests on failure counts, and Spearman's rho."""

from math import comb


def fisher_less(a, n1, b, n2):
    """P(failures in group 1 <= a) given a + b failures in n1 + n2 runs: group 1 fails less."""
    k, n = a + b, n1 + n2
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(max(0, k - n2), a + 1)) / comb(n, k)


def fisher_greater(a, n1, b, n2):
    """P(failures in group 1 >= a) given a + b failures in n1 + n2 runs: group 1 fails more."""
    k, n = a + b, n1 + n2
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(a, min(n1, k) + 1)) / comb(n, k)


def ranks(x):
    """Ranks from 1, ties sharing their mean rank."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    """Spearman's rho: the Pearson correlation of the ranks."""
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
