import numpy as np
from PIL import Image, ImageOps

class PasteOnBlack(object):
	'''
	Pastes the image on a black image
	Or pads an image with black around
	'''
	def __init__(self, max_width = 400, max_height = 300):
		self.max_width = max_width
		self.max_height = max_height
		self.max_value = max([self.max_width, self.max_height])
		
	def __call__(self, image):
		"""
		Args:
			img (PIL Image): Image to be padded

		Returns:
			PIL Image: padded Image
		"""
		width, height = image.size
		if width < self.max_width or height < self.max_height:
			width_border = (self.max_value - width) // 2
			height_border = (self.max_value - height) // 2
			image = ImageOps.expand(image, border = (width_border, height_border))
		image = image.resize((self.max_value, self.max_value))
		image = image.convert("RGB")
		return image

	def __repr__(self):
		return self.__class__.__name__ + '(%d, %d)'%(self.max_width, self.max_height)
	
class RandomizeChannels(object):
	'''
	Pastes the image on a black image
	Or pads an image with black around
	'''
	def __init__(self, p):
		self.channels = [0,1,2]
		self.p = p
		self.items = list(range(int(1 / self.p)))
		self.yes = self.items[:int(len(self.items) * self.p)]
		
	def __call__(self, image):
		"""
		Args:
			img (PIL Image): input Image

		Returns:
			rearranged channels
		"""
		if np.random.choice(self.items) in self.yes:
			np.random.shuffle(self.channels)
			return image[self.channels]
		else:
			return image

	def __repr__(self):
		return self.__class__.__name__