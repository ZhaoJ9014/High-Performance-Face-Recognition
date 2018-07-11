import scipy.io as sio 
import numpy as np 
import time
from sklearn.decomposition import PCA

print('Reading data...')
f = sio.loadmat('baseall.mat')
rt = np.float32(f['data'])
# rt = rt.transpose()
f2 = sio.loadmat('novelall.mat')
lt = np.float32(f2['data'])
# ttl = np.concatenate([rt,lt],axis=0)
rt = rt/np.linalg.norm(rt,axis=1,keepdims=True)
lt = lt/np.linalg.norm(lt,axis=1,keepdims=True)
print('PCA...')
pca = PCA(n_components=1024).fit(rt)
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

sio.savemat('novel1testall',{'data':result,'scr':scr})