class SegmentationDataset(torch.utils.data.Dataset):
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
		path_image, index_image = self.images[index]
		path_label, index_label = self.labels[index]
		image = torch.tensor(self.loader(path_image,header=None).values,dtype = torch.float)
		label = torch.tensor(self.loader(path_label,header=None).values)
		image_cuda = image.to(device)
		label_cuda = label.to(device)
		if self.transform_image:
			image_cuda = self.transform_image(image_cuda)
		if self.transform_label:
			label_cuda = self.transform_label(label_cuda)
		return image_cuda,label_cuda
	def __len__(self):
		return len(self.images)
