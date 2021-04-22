import numpy as np
import os
a = np.random.randint(0, 256, (80, 300, 400, 3)).astype('uint8')
b = np.random.randint(0, 256, (80, 300, 400, 3)).astype('uint8')
os.makedirs('example')
np.save('example/xx#PLAX-2D.npy', a)
np.save('example/xx#A4C-2D.npy', b)