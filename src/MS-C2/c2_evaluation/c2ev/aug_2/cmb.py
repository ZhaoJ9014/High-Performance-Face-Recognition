import scipy.io as sio 
import numpy as np 

f1 = sio.loadmat('base.mat')
a1 = np.float32(f1['data'])
b1 = list(f1['label'])

f2 = sio.loadmat('novel1.mat')
a2 = np.float32(f2['data'])
b2 = list(f2['label'])

a = np.append(a1,a2,axis=0)
b = b1+b2
print(a.shape)
print(len(b))

sio.savemat('base1',{'data':a,'label':b})