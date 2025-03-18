import numpy as np
import os

from src.instance import instance_reader
from src.util import constants


n = 15
dmax = 5
m = instance_reader.generate(n=n, dmax=dmax)

# Plain text format
plname = f"{n}_{dmax}_1.dat"
plpath = os.path.join(constants.extra_instance_path, plname)
f = open(plpath, "w+")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        f.write(f"{m[i][j]} ")
    f.write("\n")
f.close()

# Numpy format
npname = f"{n}_{dmax}_1.npy"
nppath = os.path.join(constants.extra_instance_path, npname)
np.save(nppath, m)

# Check
npread = np.load(nppath)
plread = np.zeros((n, n))
with open(plpath, "r") as f:
    i = 0
    for line in f.readlines():
        spl = line.split(" ")
        for j in range(len(spl)):
            if spl[j] != '\n':
                plread[i][j] = float(spl[j])
    i += 1

assert m.all() == npread.all()
assert m.all() == plread.all()
