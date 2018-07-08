import mxnet as mx
import scipy.io as sio
from skimage import io, transform
import numpy as np
import argparse,os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

def ch_dev(arg_params, aux_params, devs):
	new_args = dict()
	new_auxs = dict()
	for k, v in arg_params.items():
		new_args[k] = v.as_in_context(devs)
	for k, v in aux_params.items():
		new_auxs[k] = v.as_in_context(devs)
	return new_args, new_auxs


def main():
	kv = mx.kvstore.create(args.kv_store)
	devs = mx.cpu() if args.gpus is None else mx.gpu(args.gpus)
	begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
	model_prefix = "{}/resnext-{}-{}-{}".format(args.load_model_prefix, args.data_type, args.depth, kv.rank)
	#model_prefix = "{}/resnet-{}-{}-{}".format(args.load_model_prefix, args.data_type, args.depth, kv.rank)
	model = mx.model.FeedForward.load(model_prefix, args.model_load_epoch)
	arg_params, aux_params = ch_dev(model.arg_params, model.aux_params, devs)
	all_layers = model.symbol.get_internals()
	print model.symbol.list_arguments()
	print model.symbol.list_outputs()
	feat_symbol = all_layers['flatten0_output']

	extract = mx.io.ImageRecordIter(
		path_imgrec        = os.path.join(args.data_dir, args.rec),
		label_width        = 1,
		data_name          = 'data',
		label_name         = 'softmax_label',
		batch_size         = args.batch_size,
		data_shape         = (3, 32, 32) if args.data_type=="cifar10" else (3, args.shape, args.shape),
		rand_crop          = False,
		rand_mirror        = False,
		shuffle            = False,
		num_parts          = kv.num_workers,
		preprocess_threads = args.threads,
		part_index         = kv.rank)

	feature_extractor = mx.model.FeedForward(
		ctx                = devs,
		symbol             = feat_symbol,
		arg_params         = arg_params,
		aux_params         = aux_params,
		begin_epoch        = begin_epoch,
		allow_extra_params = True)

	features = feature_extractor.predict(extract)
	print(features.shape)
	# if  args.rec  == "Gallery_h_flip.rec":
	#     sio.savemat('{}/features_resnext_gallery_h_flip'.format(args.data_dir),{"gallery_features_resnext_s":features})
	# elif args.rec == "Gallery.rec":
	#     sio.savemat('{}/features_resnext_gallery'.format(args.data_dir),{"gallery_features_resnext_s":features})
	# elif args.rec == "Probe_h_flip.rec":
	#     sio.savemat('{}/features_resnext_probe_h_flip'.format(args.data_dir),{"probe_features_resnext_s":features})
	# elif args.rec == "Probe.rec":
	#     sio.savemat('{}/features_resnext_probe'.format(args.data_dir),{"probe_features_resnext_s":features})
	# elif args.rec == "test_h_flip.rec":
	#     sio.savemat('{}/features_resnext_test_h_flip'.format(args.data_dir),{"test_features_resnext_s":features})
	# elif args.rec == "test.rec":
	#     sio.savemat('{}/features_resnext_test'.format(args.data_dir),{"test_features_resnext_s":features})
	# elif args.rec == "train_224_h_flip.rec":
	#     sio.savemat('{}/features_resnext_train_h_flip'.format(args.data_dir),{"train_features_resnext_s":features})
	# elif args.rec == "train_224.rec":
	# sio.savemat('{}/features_resnext_train'.format(args.data_dir),{"train_features_resnext_s":features})
	# sio.savemat('extracted_feature/MSchallenge2BaseFeatureFull',{"train_features_resnext_s":features})

	name = 'challenge2'
	path = 'extracted_feature/' + name + '_feature_batch/'
	if not os.path.isdir(path):
		os.mkdir(path)

	# try:
	# 	chunk = 5000
	# 	maxIter = int(len(features)/chunk)
	# 	for iter in range(maxIter):
	# 		batch = features[iter * chunk : (iter + 1) * chunk]
	# 		print "iter_" + str(iter), " ", batch.shape
	# 		np.savetxt(path + name + '_feature_batch' + str(iter) +'.txt', batch)
	# 		# print line.shape
	# 		# print line
	# 	batch = features[(iter + 1) * chunk :]
	# 	print "iter_" + str(iter + 1), " ", batch.shape
	# 	np.savetxt(path + name + '_feature_batch' + str(iter + 1) +'.txt', batch)
	#
	# except Exception as e:
	# 	print e
	# 	try:
	# 		print "try: np.savetxt(path + name + '_Feature.txt', features)"
	# 		np.savetxt(path + name + '_Feature.txt', features)
	# 	except Exception as e:
	# 		print e
	# 		print "pickle.dump( features, open( 'extracted_feature/' + name + '_Feature.p', 'wb' ) )"
	# 		pickle.dump( features, open( 'extracted_feature/' + name + '_Feature.p', "wb" ) )

	# name = 'lowshotImg_cropped_224_test'
	np.savetxt(path + name + '_Feature.txt', features)

	# else:
	#     raise ValueError("do not support {} yet".format(args.rec))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="command for extracting features from resnet-v2")
	parser.add_argument('--gpus', type=int, default=0, help='the gpu id used for extracting feature')
	parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the extracting data directory')
	parser.add_argument('--rec', type=str, default='train_224.rec', help='the data for extracting feature')
	parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type of trained model')
	parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
	parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
	parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
	parser.add_argument('--model-load-epoch', type=int, default=0, help='load the model on an epoch using the model-load-prefix')
	parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu memory is oom, then you can try smaller value, such as --workspace 256')
	parser.add_argument('--load-model-prefix', type=str, default='./model', help='the path of model')
	parser.add_argument('--data-shape', type=int, default=256, help='the size of input data')
	parser.add_argument('--shape', type=int, default=224, help='the size of cropped data')
	parser.add_argument('--threads', type=int, default=4, help='number of thread to do preprocessing')
	args = parser.parse_args()
	main()
