import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 

gndTruth = list(sio.loadmat('noveltest.mat')['label'])

lblist = list(sio.loadmat('base1.mat')['label'])

predresult = []
prednum = np.array(sio.loadmat('novel1testresrdcp.mat')['data'])[0]
scr = np.array(sio.loadmat('novel1testresrdcp.mat')['scr'])[0]
for i in range(len(prednum)):
	predresult.append(lblist[prednum[i]])

TF = []
sc = []
# fout = open('wrong.txt','w')
for i in range(len(prednum)):
	# print(predresult[i],gndTruth[i])
	# input()
	# print(i)
	if predresult[i]==gndTruth[i]:
		TF.append(1)
	else:
		TF.append(0)
		# fout.write(gndTruth[i]+'\t'+predresult[i]+'\t'+str(scr[i])+'\n')
	sc.append(scr[i])
# fout.close()
TF = np.float32(TF)
avg = np.mean(TF)
print('Accuracy:',avg)
total = list(zip(TF,sc))
total = sorted(total,key=lambda x: x[0],reverse=True)
srt = sorted(total,key=lambda x: x[1],reverse=True)

print('Last score',srt[-1][1])
print('First score',srt[0][1])

laa = []
truenumber = 0
sss = 0
for i in range(len(prednum)):
	truenumber += srt[i][0]
	laa.append(float(truenumber)/(i+1))
for i in range(len(prednum)-1):
	if laa[i]>0.99 and laa[i+1]<0.99:
		sss = float(i)/len(prednum)
print(len(prednum))
print('Acc:',truenumber/len(prednum))
print('Cov@P=0.99:',sss)
plt.ion()
plt.plot(np.array(list(range(len(laa))))/float(len(laa)),laa)
plt.ylim(0.99,1)
plt.xlim(0.9,1)
plt.show()
plt.grid(True)
input()
