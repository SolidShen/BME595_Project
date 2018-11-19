import torchvision.datasets.folder as fold
import pandas as pd

class SegmentationDatasetBDCLSTM(torch.utils.data.Dataset):
	def __init__(self, image_dir, label_dir, train, transform_image, transform_label,loader=pd.read_csv):
		self.image_dir = image_dir
		self.label_dir = label_dir
		self.transform_image = transform_image
		self.transform_label = transform_label
		self.loader = loader
		self.classes_image, self.classes_idx_image = fold.find_classes(self.image_dir)
		self.classes_label, self.classes_idx_label = fold.find_classes(self.label_dir)
		self.images = fold.make_dataset(self.image_dir, self.classes_idx_image,'.csv')
		self.labels = fold.make_dataset(self.label_dir, self.classes_idx_label,'.csv')
		if len(self.images) == 0:
			raise(RuntimeError("Found 0 files in subfolders of: " + self.image_dir + "\n"
                              	           "Supported extensions are: " + ",".join(["CSV"])))
		if len(self.labels) == 0:
			raise(RuntimeError("Found 0 files in subfolders of: " + self.label_dir + "\n"
                                           "Supported extensions are: " + ",".join(["CSV"])))
		if len(self.images) != len(self.labels):
			raise(RuntimeError("The images and labels are not paired."))
	def __getitem__(self, index):
		if index == 0:
			index = index + 1
		elif index == len(self.images):
			index = index -1
		path_image, index_image = self.images[index]
		path_image_1, index_image_1 = self.images[index-1]
		path_image_3, index_image_3 = self.images[index+1]
		path_label, index_label = self.labels[index]
		path_label_1, index_label_1 = self.labels[index-1]
		path_label_3, index_label_3 = self.labels[index+1]
		
		image = torch.tensor(self.loader(path_image,header=None).values,dtype = torch.float)
		image_1 = torch.tensor(self.loader(path_image_1,header=None).values,dtype = torch.float)
		image_3 = torch.tensor(self.loader(path_image_3,header=None).values,dtype = torch.float)
		label = torch.tensor(self.loader(path_label,header=None).values)
		label_1 = torch.tensor(self.loader(path_label_1,header=None).values)
		label_3 = torch.tensor(self.loader(path_label_3,header=None).values)
		image_cuda = image.to(device)
		image_1_cuda = image_1.to(device)
		image_3_cuda = image_3.to(device)
		label_cuda = label.to(device)
		label_1_cuda = label_1.to(device)
		label_3_cuda = label_3.to(device)
		
		if self.transform_image:
			image_cuda = self.transform_image(image_cuda)
			image_1_cuda = self.transform_image(image_1_cuda)
			image_3_cuda = self.transform_image(image_3_cuda)
		if self.transform_label:
			label_cuda = self.transform_label(label_cuda)
			label_1_cuda = self.transform_label(label_1_cuda)
			label_3_cuda = self.transform_label(label_3_cuda)
		return image_cuda,image_1_cuda,image_3_cuda,label_cuda,label_1_cuda,label_3_cuda
	def __len__(self):
		return len(self.images)
