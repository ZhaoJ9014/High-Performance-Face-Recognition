def getDict():
	f = open('baseid120k.txt')
	d = {}
	for i in f:
		num = int(i.split(' ')[0])
		lb = i.strip().split(' ')[1]
		d[num] = lb
	print('ID list length:',len(d))
	return d

def getLabel(fname):
	f2list = []
	f = open(fname)
	for i in f:
		aa = i.replace('\n','').split('\\')[-1].replace('.jpg','').replace('.JPG','')
		f2list.append(aa)
	f.close()
	return f2list

def getFname(fname):
	f2list = []
	f = open(fname)
	for i in f:
		aa = i.replace('\n','').split('\\')[-1]
		f2list.append(aa)
	f.close()
	return f2list