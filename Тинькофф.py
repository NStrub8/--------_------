'''import math

N = int(input())
s = input().split()
t=set()
if len(s) == N:

    A= list(map(int, s))
    print(A)
    for i in range(len(A)):
        if math.floor(A[i] / 2) not in t:
            A[i] = math.floor(A[i] / 2)
            t.add(A[i])
        else: 
            t.add(A[i])
print(len(t))
print(t)

import math

N = int(input())
branches = []
for _ in range(N):
    t_i, s_i = map(int, input().split())
    branches.append((t_i, s_i))

Q = int(input())
for _ in range(Q):
    k, T = map(int, input().split())
    t_k, s_k = branches[k - 1]  
    
    if T <= t_k:
        print(t_k)
    else:
        x = math.ceil((T - t_k) / s_k)
        print(x)
        next_train = t_k + x * s_k
        print(next_train)

def count_interesting_subarrays(arr):
    n = len(arr)
    count = 0

    # Перебираем все возможные подотрезки длиной от 3 до n
    for i in range(n):
        for j in range(i + 2, n):
            subarray = arr[i:j + 1]
            found = False
            # Перебираем все тройки в пределах подотрезка
            for a in range(len(subarray)):
                for b in range(a + 1, len(subarray)):
                    for c in range(b + 1, len(subarray)):
                        if 2 * subarray[b] == subarray[a] + subarray[c]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                count += 1

    return count

# Чтение входных данных
n = int(input())
arr = list(map(int, input().split()))

# Вывод результата
print(count_interesting_subarrays(arr))


import math

N = int(input())
s = input().split()
t=set()
if len(s) == N:

    A= list(map(int, s))
    for i in range(len(A)):
        if math.floor(A[i] / 2) not in t:
            A[i] = math.floor(A[i] / 2)
            t.add(A[i])
        else: 
            t.add(A[i])
print(len(t))

from collections import deque

def is_valid(s):
    balance = 0
    for ch in s:
        if ch == '(':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0

def min_cost_to_balance(a,b,s):
    visited = set()
    queue = deque()
    queue.append((s, 0))  # (current_string, cost)
    visited.add(s)

    min_total_cost = float('inf')

    while queue:
        current, cost = queue.popleft()

        # Early stop: found valid with lower cost
        if is_valid(current):
            min_total_cost = min(min_total_cost, cost)
            continue

        # 1. Try replacements
        for i in range(len(current)):
            flipped = current[:i] + (')' if current[i] == '(' else '(') + current[i+1:]
            if flipped not in visited:
                visited.add(flipped)
                queue.append((flipped, cost + b))

        # 2. Try swaps
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                if current[i] != current[j]:  # swapping different chars is useful
                    swapped = list(current)
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    swapped_str = ''.join(swapped)
                    if swapped_str not in visited:
                        visited.add(swapped_str)
                        queue.append((swapped_str, cost + a))

    return min_total_cost

# Пример

n, a, b = map(int, input().split())
input_str = input()
print(min_cost_to_balance(a, b,input_str))  # Должно вывести 7
'''
from math import gcd
from itertools import product
from functools import reduce

MOD = 998244353

def all_beauties(a_i):
    """Находит все (p, q) такие, что p*q = a_i и НОД(p, q) = 1"""
    beauties = []
    for p in range(1, int(a_i ** 0.5) + 1):
        if a_i % p == 0:
            q = a_i // p
            if gcd(p, q) == 1:
                beauties.append((p, q))
                if p != q:
                    beauties.append((q, p))
    return beauties

def solve(n, a):
    dp = {}

    # начнем с последнего элемента последовательности
    # b[n] может быть от 1 до разумного предела (возьмем 100, можно увеличить при необходимости)
    max_start = 100
    total = 0

    for last in range(1, max_start + 1):
        stack = [([last], n - 2)]  # ([b_n, ..., b_k], index of a[index])

        while stack:
            seq, idx = stack.pop()

            if idx < 0:
                full_seq = seq[::-1]
                if gcd_all(full_seq) == 1:
                    prod = 1
                    for x in full_seq:
                        prod = (prod * x) % MOD
                    total = (total + prod) % MOD
                continue

            b_next = seq[-1]
            for p, q in all_beauties(a[idx]):
                if (b_next * p) % q == 0:
                    b_i = (b_next * p) // q
                    stack.append((seq + [b_i], idx - 1))

    return total

def gcd_all(lst):
    return reduce(gcd, lst)

# Пример:
n = int(input())
a = list(map(int, input().split()))
print(solve(n, a))  # Должно вывести 712
