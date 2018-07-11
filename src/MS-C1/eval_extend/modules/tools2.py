number = 500

def outputFile(fname,f2list,argres,scrres):
	fout = open(fname,'w')
	for i in search:
		fout.write(f2list[i]+','+d[argres[i]]+','+str(scrres[i])+','+str(int(d[argres[i]]==f2list[i]))+'\n')
	fout.close()

def getSorted(truthlist,scrres):
	total = list(zip(truthlist,scrres))
	total = sorted(total,key=lambda x: x[0],reverse=True)
	res = sorted(total,key=lambda x: x[1],reverse=True)
	return res

def getTruthlist(lbs,f2list):
	total = 0
	res = []
	assert len(lbs)==len(f2list)
	for i in range(len(lbs)):
		if lbs[i]==f2list[i]:
			res.append(1)
			total+=1
		else:
			res.append(0)
	print('Accuracy:',total/len(f2list))
	return res

def setnumber(nb):
	global number
	number = nb

def plot(srt,ion=True):
	import matplotlib.pyplot as plt 
	import numpy as np 
	truenumber = 0
	laa = []
	for i in range(number):
		truenumber += srt[i][0]
		laa.append(float(truenumber)/(i+1))

	cvg99 = 0
	cvg95 = 0
	scr95 = 0
	for i in range(number-1):
		if laa[i]>=0.95 and laa[i+1]<0.95:
			print('95 pos:',i)
			cvg95=float(i)/float(number)
			scr95 = srt[i][1]
		if laa[i]>=0.99 and laa[i+1]<0.99:
			print('99 pos:',i)
			cvg99=float(i)/float(number)

	print('cvg@P99:',cvg99)
	print('cvg@P95:',cvg95)
	print('scr95:',scr95)
	if ion:
		plt.ion()
	plt.plot(np.array(list(range(len(laa))))/number,laa)
	plt.ylim(0.95,1)
	plt.xlim(0.5,0.7)
	plt.grid(True)
	plt.show()
	

def getLabel(d,argres):
	lbs = []
	for i in range(len(argres)):
		lbs.append(d[argres[i]])
	return lbs

def sortTable(gtruth,pred,scrres,truthlist):
	a = list(range(number))
	total = list(zip(gtruth,pred,scrres,truthlist))
	total = sorted(total,key=lambda x:x[3],reverse=True)
	srt = sorted(total,key=lambda x:x[2],reverse=True)
	srt = [(a[i],srt[i][0],srt[i][1],srt[i][2],srt[i][3]) for i in range(number)]
	return srt

def getArgAndScore(slist,tlist):
	import numpy as np 
	sall = np.zeros(slist[0].shape)
	tall = np.zeros(tlist[0].shape)
	# sall = np.add(slist)
	# tall = np.add(tlist)
	for i in range(len(slist)):
		sall+=slist[i]
		tall+=tlist[i]
	tall[tall==0]=-1.0
	tall = 1/tall
	tall[tall<0]=0.0
	sall = sall*tall
	argres = np.argmax(sall,axis=1)
	scrres = list(np.max(sall,axis=1))
	return argres,scrres

def getTop5ArgAndScr(slist,tlist):
	import numpy as np 
	sall = np.zeros(slist[0].shape)
	tall = np.zeros(tlist[0].shape)
	# sall = np.add(slist)
	# tall = np.add(tlist)
	for i in range(len(slist)):
		sall+=slist[i]
		tall+=tlist[i]
	tall[tall==0]=-1.0
	tall = 1/tall
	tall[tall<0]=0.0
	sall = sall*tall
	argres = np.argmax(sall,axis=1)
	scrres = list(np.max(sall,axis=1))
	print('calculating top5...')
	argres5 = np.argpartition(-sall,5,axis=1)[:,:5]
	scrres5 = np.array([sall[i][argres5[i]] for i in range(len(argres5))])
	print(argres5[0])
	buff = np.argsort(-scrres5,axis=1)
	argres5 = argres5[buff]
	scrres5 = scrres5[buff]
	# scrres5 = np.zeros(argres5.shape,dtype=np.float32)
	# for i in range(len(argres5)):
	# 	for j in range(len(argres5[0])):
	# 		scrres5[i][j] = sall[i][argres5[i][j]]
	return argres,scrres,argres5,scrres5