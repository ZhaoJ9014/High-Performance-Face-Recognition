f = open('master.txt')
f1 = open('slavenoov.txt')
fout = open('transnoov.txt','w')
d = {}
cnt = 0
for i in f:
	i = i.strip()
	d[i] = cnt
	cnt+=1
f.close()
cnt = 0
for i in f1:
	i = i.strip()
	fout.write(str(cnt)+' '+str(d[i])+'\n')
	cnt+=1
fout.close()