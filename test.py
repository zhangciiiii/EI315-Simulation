from collections import Counter


x = [1, 2, 3, 3, 2, 1, 0, 2]

print(Counter(x).most_common(1)[0][0])
