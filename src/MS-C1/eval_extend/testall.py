import sys
sys.path.append('./modules/')
import checkall
import numpy as np 
from namedict import getDict,getLabel
import tools
import table

print('Preparing libraries...')
#Testing image list txt
f2list = getLabel('set1list.txt')

#get data
d = getDict()

number = len(f2list)
# number = 1385
checkall.setnumber(number)
tools.setnumber(number)

print('Reading data...')
#get score and timer metrics
# s0,t0 = checkall.getScr(['feature_vec_100k_nopca'],'transfv.txt')
# s01,t01 = checkall.getScr(['feature_res_vec_100k2_pca256'],'transfvres.txt')
# s1,t1 = checkall.getScr(['enf11_1','enf12_1','enf13_1','enf14_1','enf15_1'],'transbase.txt')
# s2,t2 = checkall.getScr(['enf11_2','enf12_2','enf13_2','enf14_2','enf15_2'],'transsf.txt')
# s3,t3 = checkall.getScr(['enf11_3','enf12_3','enf13_3','enf14_3','enf15_3'],'translayer3.txt')
# s4,t4 = checkall.getScr(['enf11_4','enf12_4','enf13_4','enf14_4','enf15_4'],'translayer4.txt')
# s5,t5 = checkall.getScr(['enf11_5','enf12_5','enf13_5','enf14_5','enf15_5'],'translayer5.txt')


s0,t0 = checkall.getScr(['feature2_vec_100k'],'transfv.txt')
s01,t01 = checkall.getScr(['feature2_res_vec_100k'],'transfvres.txt')
s1,t1 = checkall.getScr(['enf21_1','enf22_1','enf23_1','enf24_1','enf25_1'],'transbase.txt')
s2,t2 = checkall.getScr(['enf21_2','enf22_2','enf23_2','enf24_2','enf25_2'],'transsf.txt')
s3,t3 = checkall.getScr(['enf21_3','enf22_3','enf23_3','enf24_3','enf25_3'],'translayer3.txt')
s4,t4 = checkall.getScr(['enf21_4','enf22_4','enf23_4','enf24_4','enf25_4'],'translayer4.txt')
s5,t5 = checkall.getScr(['enf21_5','enf22_5','enf23_5','enf24_5','enf25_5'],'translayer5.txt')

# s1,t1 = checkall.getScr(['enf31_1','enf32_1','enf33_1','enf34_1','enf35_1'],'transbase.txt')
# s2,t2 = checkall.getScr(['enf31_2','enf32_2','enf33_2','enf34_2','enf35_2'],'transsf.txt')
# s3,t3 = checkall.getScr(['enf31_3','enf32_3','enf33_3','enf34_3','enf35_3'],'translayer3.txt')

# s0,t0 = checkall.getScr(['feature4_vec_100k'],'transfv.txt')
# s01,t01 = checkall.getScr(['feature4_res_vec_100k'],'transfvres.txt')
# s1,t1 = checkall.getScr(['enf41_1','enf42_1','enf43_1','enf44_1','enf45_1'],'transbase.txt')
# s2,t2 = checkall.getScr(['enf41_2','enf42_2','enf43_2','enf44_2','enf45_2'],'transsf.txt')
# s3,t3 = checkall.getScr(['enf41_3','enf42_3','enf43_3','enf44_3','enf45_3'],'translayer3.txt')
# s4,t4 = checkall.getScr(['enf41_4','enf42_4','enf43_4','enf44_4','enf45_4'],'translayer4.txt')
# s5,t5 = checkall.getScr(['enf41_5','enf42_5','enf43_5','enf44_5','enf45_5'],'translayer5.txt')

# s0,t0 = checkall.getScr(['feature5_vec_100k'],'transfv.txt')
# s01,t01 = checkall.getScr(['feature5_res_vec_100k'],'transfvres.txt')
# s1,t1 = checkall.getScr(['enf51_1','enf52_1','enf53_1','enf54_1','enf55_1'],'transbase.txt')
# s2,t2 = checkall.getScr(['enf51_2','enf52_2','enf53_2','enf54_2','enf55_2'],'transsf.txt')
# s3,t3 = checkall.getScr(['enf51_3','enf52_3','enf53_3','enf54_3','enf55_3'],'translayer3.txt')
# s4,t4 = checkall.getScr(['enf51_4','enf52_4','enf53_4','enf54_4','enf55_4'],'translayer4.txt')
# s5,t5 = checkall.getScr(['enf51_5','enf52_5','enf53_5','enf54_5','enf55_5'],'translayer5.txt')

print('Calculating...')
#do ensemble
argres,scrres,argt5,scrt5 = tools.getTop5ArgAndScr([s1,s2,s3,s4,s5],[t1,t2,t3,t4,t5])
arg1,scr1 = tools.getArgAndScore([s0],[t0])
arg2,scr2 = tools.getArgAndScore([s01],[t01])

print(len(scrres))
for i in range(len(argres)):
	if argres[i] == arg1[i]:
		if arg1[i]==arg2[i]:
			scrres[i] += 1.6
		else:
			scrres[i] += 0.8
	elif arg1[i] == arg2[i]:
		print(scr1[i],scr2[i])
		if (scr1[i]>0.4 or scr2[i]>0.4):
			if (scrres[i]<0.7):
				argres[i] = arg1[i]
				if argt5[i][1]==arg1[i]:
					scrres[i] = 0.85+0.001*(scr1[i]+scr2[i])
				else:
					scrres[i] = 0.8+0.001*(scr1[i]+scr2[i])
	# elif scr1[i]>0.5:
	# 	argres[i] = arg1[i]
	# 	scrres[i] = 0.82+scr1[i]

print(len(argres))
print(len(f2list))

#eval
#get max score and label
lbs = tools.getLabel(d,argres)
truthlist = tools.getTruthlist(lbs,f2list)

print('Plotting...')
srt = tools.getSorted(truthlist,scrres)
tools.plot(srt)
tablelist = tools.sortTable(f2list,lbs,scrres,truthlist)
table.getTable(tablelist)