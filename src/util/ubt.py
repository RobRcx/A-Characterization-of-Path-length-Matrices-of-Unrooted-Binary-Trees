import numpy as np
import numpy.linalg as la

from ordered_set import OrderedSet

class UBT:
    # Initializes basic UBT
    def __init__(self):
        self.n = 4
        self.adj = [[] for _ in range(4)]
        self.leaves = OrderedSet([1, 2, 3])
        self.depth = [0, 1, 1, 1]
        for i in range(3):
            self.adj[0].append(i + 1)
            self.adj[i + 1].append(0)
        self.tau = self.pldm = None

    def add_layer(self):
        removed = OrderedSet()
        added = OrderedSet()
        for l in self.leaves:
            d = self.depth[l]
            for _ in range(2):
                self.adj[l].append(self.n)
                self.adj.append([])
                self.adj[self.n].append(l)
                self.depth.append(d + 1)
                added.add(self.n)
                self.n += 1
            removed.add(l)
        for l in removed:
            self.leaves.remove(l)
        for l in added:
            self.leaves.add(l)
        print(self.leaves)

    def expand(self, leaf):
        self.leaves.remove(leaf)
        added = []
        for _ in range(2):
            self.adj[leaf].append(self.n)
            self.adj.append([])
            self.adj[self.n].append(leaf)
            self.depth.append(self.depth[leaf] + 1)
            self.leaves.add(self.n)
            added.append(self.n)
            self.n += 1
        return added

    def dfs(self, v, p, h, d):
        if v != p and len(self.adj[v]) == 1:
            d[v] = h
            return
        for u in self.adj[v]:
            if u != p:
                self.dfs(u, v, h + 1, d)

    def check_PLM_PLDM(self):
        if self.tau is None or (len(self.leaves) != len(self.tau)):  # TODO: strengthen check
            return False
        return True

    def compute_PLM_PLDM(self):
        if self.check_PLM_PLDM():
            return self.tau, self.pldm
        c = 0
        ltr = {}
        for l in self.leaves:
            ltr[l] = c
            c += 1
        self.tau = [[0 for _ in range(c)] for _ in range(c)]
        self.pldm = [[0 for _ in range(c)] for _ in range(c)]
        for l in self.leaves:
            d = {}
            self.dfs(l, l, 0, d)
            for k, v in d.items():
                self.tau[ltr[l]][ltr[k]] = self.tau[ltr[k]][ltr[l]] = v
                self.pldm[ltr[l]][ltr[k]] = self.pldm[ltr[k]][ltr[l]] = 1 / 2 ** v
        return self.tau, self.pldm

    @staticmethod
    def spectrum(mat, digits=6):
        tmp = mat
        if type(tmp) is not np.array:
            tmp = np.array(tmp)
        evaltau, evectau = la.eigh(np.array(tmp))
        evectau_t = []
        for i in range(len(evectau)):
            x = [0 for _ in range(len(evectau[:]))]
            for j in range(len(evectau[:])):
                x[j] = round(evectau[j, i], digits)
            evectau_t.append(x)
        evectau_t = np.array(evectau_t)
        return evaltau, evectau_t

    def spectrum_PLM(self, digits=6):
        if not self.check_PLM_PLDM():
            self.compute_PLM_PLDM()
        return self.spectrum(self.tau, digits)

    def spectrum_PLDM(self, digits=6):
        if not self.check_PLM_PLDM():
            self.compute_PLM_PLDM()
        return self.spectrum(self.pldm, digits)