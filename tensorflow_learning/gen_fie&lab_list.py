#generate picture&label list
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import os

MODE = 'folder' # 可选'file' 一半用于文本信息
DATASET_PATH = r'./testdata' # 数据集根目录

N_CLASSES = 2 # 有几类（多少个子文件夹）
IMG_HEIGHT = 64 # (预处理图像高度)
IMG_WIDTH = 64 # (预处理图像宽度)
CHANNELS = 3 # 如果是灰度图则为1

def read_images(dataset_path, mode):#传入根目录，模式
	imagepaths, labels = list(), list()
	dict={}
	if mode == 'file':
		# Read dataset file
		data = open(dataset_path, 'r').read().splitlines()
		for d in data:
			imagepaths.append(d.split(' ')[0])
			labels.append(int(d.split(' ')[1]))
	elif mode == 'folder':
		# An ID will be affected to each sub-folders by alphabetical order
		label = 0
		# List the directory
		
		try:  # Python 2
			classes = sorted(os.walk(dataset_path).next()[1])
			
		except Exception:  # Python 3
			classes = sorted(os.walk(dataset_path).__next__()[1])
		
		#得到类别，文件夹名字
		# List each sub-directory (the classes)
		for c in classes:
			c_dir = os.path.join(dataset_path, c)#拼成子文件夹全名
			try:  # Python 2
				walk = os.walk(c_dir).next()
			except Exception:  # Python 3
				walk = os.walk(c_dir).__next__()
			# Add each image to the training set
			for sample in walk[2]:#遍历这个子文件夹里面的图片
				# Only keeps jpeg images
				if sample.endswith('.png') or sample.endswith('.png'):
					imagepaths.append(os.path.join(c_dir, sample))#增加一个图的地址到list1
					labels.append(label)#把对应的label加入到第二个list2(这里可以自己拼路径和label)
			#拼完一个类到list之后
			dict.update({c:label})
			#print('拼出来的字典',dict)#{'trainingJunk_png': 0, 'trainingSymbols_png': 1}
			label += 1
	else:
		raise Exception("Unknown mode.")
	return imagepaths,labels,dict#返回图片全名list和label_list 以及字典




























