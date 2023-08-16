import numpy as np

with open('recognize-digit/src/weights.ts', 'w') as file:
    file.write('export const w1 =' + str(np.load('w1.npy').tolist()))
    file.write('\nexport const w2 =' + str(np.load('w2.npy').tolist()))
