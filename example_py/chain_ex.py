from itertools import chain

a = [1, 2, 3]
b = ['a', 'b']
c = (10, 20)

for x in chain(a, b, c):
    print(x)
