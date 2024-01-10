import torch.nn as nn

class VGG11(nn.Module):
	def __init__(self, input_channel, n_classes, image_resolution, VGGArchitecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
		super().__init__()
		self.input_channel = input_channel

		def VGGBlock(num_convs, input_channel, output_channel):
			layers = []
			for _ in range(num_convs):
				layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1))
				layers.append(nn.ReLU())
				input_channel = output_channel

			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			return nn.Sequential(*layers)

		conv_blcks = []		
		for (num_convs, output_channel) in VGGArchitecture:
			conv_blcks.append(VGGBlock(num_convs, self.input_channel, output_channel))
			self.input_channel = output_channel

		self.layers = nn.Sequential(
			*conv_blcks, 
			nn.Flatten(),
			nn.Linear(output_channel * (image_resolution//(2**len(VGGArchitecture))) * (image_resolution//(2**len(VGGArchitecture))), 4096),
			nn.ReLU(), nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(), nn.Dropout(0.5),
			nn.Linear(4096, n_classes))

		self.layers.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
			nn.init.normal_(layer.weight)

	def forward(self, x):		
		out = self.layers(x)
		return out
