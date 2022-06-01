
import matplotlib.cm as cm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import h5py

#izpi posnetka slik
f2 = h5py.File('data.hdf5', 'r')

n1 = f2.get('slike')

n1 = np.array(n1)

frames = [] # for storing the generated images
#fig = plt.figure()


#for i in range(len(n1)):
    #frames.append([plt.imshow(n1[i], cmap=cm.Greys_r,animated=True)])

#ani = animation.ArtistAnimation(fig, frames, interval=250, blit=True, repeat_delay=1000)
#plt.show()



#izpis ostalih vsebin hdf5
#fig, (ax1, ax2, ax3) = plt.subplots(3)

hitrosti = f2['hitrosti']
zavora = f2['zavora']
volan = f2['volan']

data1 = hitrosti[:]
data2 = zavora[:]
data3 = volan[:]





plt.figure()

plt.subplot(3, 1, 1)
plt.plot(data1)
plt.title('Hitrosti')



plt.subplot(3, 1, 2)
plt.plot(data2)
plt.title('Zavora')



plt.subplot(3, 1, 3)
plt.plot(data3)
plt.title('Volan')


plt.tight_layout()

plt.show()


'''
plt.title("Hitrosti")
ax1.plot(data1)
plt.title("Hitrosti")
ax2.plot(data2)
plt.title("Hitrosti")
ax3.plot(data3)
plt.show()
'''
