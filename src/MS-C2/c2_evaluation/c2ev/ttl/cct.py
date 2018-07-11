import scipy.io as sio 
import numpy as np 

# f = sio.loadmat('base1')['data']
# a = np.array(f)
# a1 = np.zeros(a.shape,dtype=np.float32)
# f = open('transres50.txt')
# d = {}
# for i in f:
# 	i = i.strip().split(' ')
# 	i0 = int(i[0])
# 	i1 = int(i[1])
# 	d[i0] = i1
# f.close()
# for k in d:
# 	a1[d[k]] = a[k]
lb = sio.loadmat('noveltest')['label']
lb = np.array(lb)
f = sio.loadmat('noveltest1')['data']
a1 = np.array(f)
f = sio.loadmat('noveltest')['data']
a0 = np.array(f)
a1 = np.concatenate([a0,a1],axis=1)
sio.savemat('novelall',{'data':a1,'label':lb})