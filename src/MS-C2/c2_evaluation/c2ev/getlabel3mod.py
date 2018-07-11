import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 

gndTruth = list(sio.loadmat('noveltest.mat')['label'])

lblist = list(sio.loadmat('base1.mat')['label'])

predresult = []

# prednum = np.array(sio.loadmat('novel1testres6_2.mat')['data'])[0]
# scr = np.array(sio.loadmat('novel1testres6_2.mat')['scr'])[0]

prednum2 = np.array(sio.loadmat('novel1testresnoov2.mat')['data'])[0]
scr2 = np.array(sio.loadmat('novel1testresnoov2.mat')['scr'])[0]
prednum3 = np.array(sio.loadmat('novel1testres6_2.mat')['data'])[0]
scr3 = np.array(sio.loadmat('novel1testres6_2.mat')['scr'])[0]
# prednum3 = np.array(sio.loadmat('novel1testdense_0714.mat')['data'])[0]
# scr3 = np.array(sio.loadmat('novel1testdense_0714.mat')['scr'])[0]
# prednum4 = np.array(sio.loadmat('novel1testgbnzj_0714.mat')['data'])[0]
# scr4 = np.array(sio.loadmat('novel1testgbnzj_0714.mat')['scr'])[0]
prednum = np.array(sio.loadmat('novel1testdensegbn_0716.mat')['data'])[0]
scr = np.array(sio.loadmat('novel1testdensegbn_0716.mat')['scr'])[0]

prednum1k = np.array(sio.loadmat('novel1testres1k2_noov.mat')['data'])[0]
scr1k = np.array(sio.loadmat('novel1testres1k2_noov.mat')['scr'])[0]

scr = scr*10
print(scr.shape)
for i in range(len(prednum)):
	if prednum[i]==prednum2[i]:
		#1=2
		scr[i] += 2
		if prednum[i]==prednum3[i]:
			#1=2=3
			scr[i] += 2
		# elif prednum[i]==prednum4[i]:
		# 	#1=2=4
		# 	scr[i] += 2
	elif prednum[i]==prednum3[i]:
		#1=3
		scr[i] += 2
		# if prednum[i]==prednum4[i]:
		# 	#1=3=4
		# 	scr[i] += 0
	# elif prednum[i]==prednum4[i]:
	# 	scr[i] += 0
	# elif prednum2[i]==prednum3[i]:
	# 	if prednum2[i]==prednum4[i]:
	# 		prednum[i] = prednum2[i]
	# 		scr[i] = 1 + (scr2[i]+scr3[i]+scr4[i])/3
	else:
		# scr[i] -= 2
		if prednum[i]!=prednum1k[i] and scr1k[i]>0.3:
			scr[i] = 1+scr1k[i]
			prednum[i] = prednum1k[i]
		if prednum[i] == prednum1k[i]:
			scr[i] += 2
	if prednum[i]!=prednum1k[i] or (scr[i]<3.5 and scr[i]>3) or (scr[i]<1.5 and scr[i]>1):
		scr[i] -= 10
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
# print('Negative right:',negative1)
# print('Negative wrong:',negative2)
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
sss = 1.0
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
print('Threshold at 0.99:',thre)
plt.ion()
plt.plot(np.array(list(range(len(laa))))/float(len(laa)),laa)
plt.ylim(0.99,1)
plt.xlim(0.9,1)
plt.show()
plt.grid(True)
input()
