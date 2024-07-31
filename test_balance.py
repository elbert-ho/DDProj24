import numpy as np
import matplotlib.pyplot as plt
qed = np.load("data/qed.npy")
plt.hist(qed)
plt.savefig("imgs/qed.png")
sas = np.load("data/sas.npy")
plt.clf()
plt.hist(sas, bins=20)
plt.savefig("imgs/sas.png")

pIC50 = np.load("data/pIC50.npy")
plt.clf()
plt.hist(pIC50, bins=20)
plt.savefig("imgs/z_pIC50.png")