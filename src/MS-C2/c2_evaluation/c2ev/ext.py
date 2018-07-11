import scipy.io as sio 
import numpy as np 

f = sio.loadmat('base1')['label']
lb = list(f)
f = open('master.txt','w')
for i in lb:
	f.write(i.replace(' ','')+'\n')
f.close()