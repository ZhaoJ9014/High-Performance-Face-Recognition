import scipy.io as sio 
import numpy as np 

f = sio.loadmat('novel1')['label']
lb = list(f)
f = open('slave1k.txt','w')
for i in lb:
	f.write(i.replace(' ','')+'\n')
f.close()