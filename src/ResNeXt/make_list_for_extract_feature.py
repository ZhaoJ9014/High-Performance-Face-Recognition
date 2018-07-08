import argparse,os

def main():

  if args.data_name == "train":
      if args.h_flip == 1:
        with open('{}/split{}/train_224_h_flip.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/train_224_h_flip.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
      else:
        with open('{}/split{}/train_224.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/train_224.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
  elif args.data_name == "test":
      if args.h_flip == 1:
        with open('{}/split{}/test_h_flip.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/test_h_flip.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
      else:
        with open('{}/split{}/test.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/test.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
  elif args.data_name == "Gallery":
      if args.h_flip == 1:
        with open('{}/split{}/Gallery_h_flip.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/Gallery_h_flip.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
      else:
        with open('{}/split{}/Gallery.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/Gallery.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
  elif args.data_name == "Probe":
      if args.h_flip == 1:
        with open('{}/split{}/Probe_h_flip.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/Probe_h_flip.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
      else:
        with open('{}/split{}/Probe.txt'.format(args.data_dir,args.split_num), 'r') as f:
            lines = f.readlines()
            f_ = open('{}/split{}/Probe.lst'.format(args.out_dir,args.split_num), 'w')
            for line in lines:
                xx = line.split('\t')
                xx[2] = xx[2].strip('\n').strip('\r')
                f_.write('%d\t%d\t%s\n' % (int(xx[1]), int(xx[2]), xx[0]))
            f_.close()
  else:
      raise ValueError("do not support {} yet".format(args.data_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command of making list for extract feature")
    parser.add_argument('--data-dir', type=str, default='./data/IJBA', help='the input data directory')
    parser.add_argument('--split-num', type=int, default=1, help='the number of split')
    parser.add_argument('--out-dir', type=str, default='./data/IJBA', help='the output data directory')
    parser.add_argument('--h-flip', type=int, default=0, help='whether horizontally flip image or not')
    parser.add_argument('--data-name', type=str, default='train', help='the data name of you want to make list')
    args = parser.parse_args()
    main()