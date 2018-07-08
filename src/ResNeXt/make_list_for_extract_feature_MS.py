import argparse,os
import pickle

IDtoNum = {}
NumtoID = {}
def main():
	path = "./"
	# name = 'baseImage_224's
	# name = "lowshotImg_cropped_224"
	# name = "DevBaseSet"
	# name = "DevNovelSet_randomCrop"
	# name = "NovelSet_1_generated"
	name = "challenge2"

	with open(path + name + '.txt', 'r') as f:
		lines = f.readlines()
		f_1 = open(name + '.lst', 'w')
		# f_1 = open('75.lst', 'w')

		IDcounter = 0
		counter = 1
		print "len(lines): ", len(lines)
		for line in lines:
			# ##################################################
			# if counter > 1155100:
			# ##################################################
			xx = line.split('\t')
			xx[1] = xx[1].strip('\n').strip('\r')

			if xx[1] in IDtoNum:
				numID = IDtoNum[xx[1]]
			else:
				IDtoNum[xx[1]] = IDcounter
				NumtoID[IDcounter] = xx[1]
				numID = IDtoNum[xx[1]]
				IDcounter += 1
				print "IDcounter: ", IDcounter

			f_1.write( str(counter) + "\t" + str(float(numID)) + "\t" + xx[0] + "\n")
			print counter

			counter += 1

		# name = "75"

		f_1.close()
		pickle.dump( IDtoNum, open( name + "_IDtoNum.p", "wb" ) )
		pickle.dump( NumtoID, open( name + "_NumtoID.p", "wb" ) )

		# f_2.close()





if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description="command of making list for extract feature")
	# parser.add_argument('--data-dir', type=str, default='./data/IJBA', help='the input data directory')
	# parser.add_argument('--split-num', type=int, default=1, help='the number of split')
	# parser.add_argument('--out-dir', type=str, default='./data/IJBA', help='the output data directory')
	# parser.add_argument('--h-flip', type=int, default=0, help='whether horizontally flip image or not')
	# parser.add_argument('--data-name', type=str, default='train', help='the data name of you want to make list')
	# args = parser.parse_args()
	main()
