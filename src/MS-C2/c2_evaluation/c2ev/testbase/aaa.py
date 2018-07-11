import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 

gndTruth = list(sio.loadmat('noveltest.mat')['label'])

lblist = list(sio.loadmat('base1.mat')['label'])

predresult = []

prednum = np.array(sio.loadmat('novel1testresrdcp2.mat')['data'])[0]
scr = np.array(sio.loadmat('novel1testresrdcp2.mat')['scr'])[0]

for i in range(len(prednum)):
	predresult.append(lblist[prednum[i]])

TF = []
sc = []
wrongnum = 0
negative1 = 0
negative2 = 0
fout = open('wrong.txt','w')
for i in range(len(prednum)):
	# print(predresult[i],gndTruth[i])
	# input()
	if predresult[i]==gndTruth[i]:
		if scr[i]<0:
			negative1+=1
		TF.append(1)
	else:
		TF.append(0)
		if scr[i]<0:
			negative2+=1
		fout.write(gndTruth[i]+'\t'+predresult[i]+'\t'+str(scr[i])+'\n')
		wrongnum+=1
	sc.append(scr[i])
fout.close()
print('Negative right:',negative1)
print('Negative wrong:',negative2)
print('Wrong:',wrongnum)
TF = np.float32(TF)
avg = np.mean(TF)
print('Accuracy:',avg)
total = list(zip(TF,sc))
# total = sorted(total,key=lambda x: x[0],reverse=True)
srt = sorted(total,key=lambda x: x[1],reverse=True)

print('Last score',srt[-1][1])
print('First score',srt[0][1])

laa = []
truenumber = 0
sss = 0
thre = 0
for i in range(len(prednum)):
	truenumber += srt[i][0]
	laa.append(float(truenumber)/(i+1))
for i in range(len(prednum)-1):
	if laa[i]>0.99 and laa[i+1]<0.99:
		print('99pos:',i)
		sss = float(i)/len(prednum)
		thre = srt[i][1]
print(len(prednum))
print('Acc:',truenumber/len(prednum))
print('Cov@P=0.99:',sss)
print('Threshold:',thre)
plt.ion()
plt.plot(np.array(list(range(len(laa))))/float(len(laa)),laa)
plt.ylim(0.99,1)
plt.xlim(0.9,1)
plt.show()
plt.grid(True)
input()
