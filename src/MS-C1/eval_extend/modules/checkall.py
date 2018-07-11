import scipy.io as sio
import numpy as np 

number = 500

def setnumber(nb):
	global number
	number = nb

def getScr(flist,labfile):
	f = open(labfile)
	l = []
	for i in f:
		i = i.strip().split(' ')
		l.append((int(i[0]),int(i[1])))
	f.close()

	ars = []
	for fname in flist:
		fname = './data/'+fname
		f = sio.loadmat(fname)['data']
		ar = np.float32(f)
		ars.append(ar)

	arall = np.concatenate(ars,axis=1)
	# print(arall.shape)
	res = np.zeros([number,120000],dtype=np.float32)
	timer = np.zeros([1,120000],dtype=np.float32)
	for i in l:
		if i[0]!=-1 and i[1]!=-1:
			res[:,i[1]] = arall[:,i[0]]
			sm = np.sum(arall[:,i[0]])
			if sm!=0:
				timer[0][i[1]] = 1.0
	# print(res.shape)
	# print(timer.shape)
	return res,timer