file = open('lfw_list.txt','r')
person_name = []
for lines in file:
	person_name.append(lines.split('/')[9].split('_')[0]+'_'+lines.split('/')[9].split('_')[1])

file.close()

unique_name = []
tmp = person_name[0]
unique_name.append(tmp)
for i in range(len(person_name)):
	if person_name[i] != tmp:
		tmp = person_name[i]
		unique_name.append(tmp)

lst_content = []

count = 0

file = open('lfw_list.txt','r')
for lines in file:
	lst_content.append(str(count) + '\t' + str(float(unique_name.index(person_name[count]))) + '\t' + lines.split(' ')[0])
	count = count + 1

file.close()

file = open('lfw_list.lst', 'w')
for i in range(len(lst_content)):
	file.write(lst_content[i]+'\n')

file.close()