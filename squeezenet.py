import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
	def __init__(self, inplanes, squeeze_planes, expand_planes):
		super(Fire, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
		self.bn1 = nn.BatchNorm2d(squeeze_planes)

		self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1)
		self.bn2 = nn.BatchNorm2d(expand_planes)

		self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(expand_planes)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))

		out1 = F.relu(self.bn2(self.conv2(x)))

		out2 = F.relu(self.bn3(self.conv3(x)))

		output = F.relu(torch.cat([out1, out2], 1))
		return output





class SqueezeNet(nn.Module):
	def __init__(self):
		super(SqueezeNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(96)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fire1 = Fire(96, 16, 64)
		self.fire2 = Fire(128, 16, 64)
		self.fire3 = Fire(128, 32, 128)
		self.fire4 = Fire(256, 32, 128)
		self.fire5 = Fire(256, 48, 192)
		self.fire6 = Fire(384, 48, 192)
		self.fire7 = Fire(384, 64, 256)
		self.fire8 = Fire(512, 64, 256)
		self.conv2 = nn.Conv2d(512, 10, kernel_size=1)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.maxpool(x)

		x = self.fire1(x)
		x = self.fire2(x)
		x = self.fire3(x)
		x = self.maxpool(x)

		x = self.fire4(x)
		x = self.fire5(x)
		x = self.fire6(x)
		x = self.fire7(x)
		x = self.maxpool(x)

		x = self.fire8(x)
		x = self.avg_pool(self.conv2(x))
		x = F.log_softmax(x, dim=1)
		return x

model = SqueezeNet()
x = torch.randn(2, 3,224,224)
y = model(x)
print(y.size())