import scipy.io as sio 
import numpy as np 
import time
from sklearn.decomposition import PCA

print('Reading data...')
f = sio.loadmat('base1.mat')
rt = np.float32(f['data'])
# rt = rt.transpose()
f2 = sio.loadmat('basetest.mat')
lt = np.float32(f2['data'])
rt = rt/np.linalg.norm(rt,axis=1,keepdims=True)
lt = lt/np.linalg.norm(lt,axis=1,keepdims=True)
print('PCA...')
pca = PCA(n_components=192)
pca = pca.fit(lt)
rt = pca.transform(rt)
lt = pca.transform(lt)
rt = rt/np.linalg.norm(rt,axis=1,keepdims=True)
rt = rt.transpose()
lt = lt/np.linalg.norm(lt,axis=1,keepdims=True)
print(rt.shape)
print(lt.shape)
print('Computing the result...')
a = time.time()
result = np.dot(lt,rt)
b = time.time()
print('time elapsed:',b-a)
print(result.shape)
scr = np.amax(result,axis=1)
result = np.argmax(result,axis=1)

#get transform dict
d = {}
f = open('transrdcp.txt')
for i in f:
	i = i.strip().split(' ')
	a = int(i[0])
	b = int(i[1])
	d[a] = b
f.close()

for i in range(len(result)):
	result[i] = d[result[i]]

sio.savemat('basetestresrdcp2',{'data':result,'scr':scr})