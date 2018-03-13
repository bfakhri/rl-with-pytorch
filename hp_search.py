import itertools
import os
def comb_lists(b, l, m, e):
     print(list(itertools.product(b, l, m, e))) 

bSize = list(range(20,1000,5))
lRate = list()
mEps = list(range(10000,100000,1000))
envs = "Pong-v0"
deci = range(1,8)

for i in deci:
    lRate.append(.1**i)

comb_lists(bSize , lRate , mEps , envs)
