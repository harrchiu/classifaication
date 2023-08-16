import numpy as np

for file in ['w1','w2']:
    np.savetxt(file + '.txt', np.load(file + '.npy'), delimiter=',', fmt='%5.7f')