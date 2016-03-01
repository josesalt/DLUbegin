import hashlib
import time
from itertools import combinations
from six.moves import cPickle as pickle
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt

root = os.curdir
print (root)

for pickleFile in os.listdir(root):
    if (pickleFile.endswith('A.pickle')):
        pickle_file = os.path.join(root, pickleFile)
        ax = pickle.load(open(pickle_file,  'rb') )

        print len(ax)
        ax.flags.writeable=False

        ax_lim= [ax[595],ax[628]]
        
##        contador = 0
        for img in ax_lim:
            h = hash(img.data)           
            h2 = hashlib.sha1(img.data).hexdigest()
            print img
            
##            contador=contador+1
##            if contador ==10:
##                break

            plt.imshow(img)
            plt.show()
            
            print (h,", ", h2)
      
##        imgRand = randint(0,len(ax))
