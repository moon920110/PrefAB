import torch.nn as nn


class FiLM(nn.Module):
	def __init__(self, d_bio, d_model):
		super(FiLM, self).__init__()
		bio_layers = []

		while d_bio * 2 < d_model:
			bio_layers.append(nn.Linear(d_bio, d_bio*2))
			bio_layers.append(nn.LayerNorm(d_bio*2))
			bio_layers.append(nn.LeakyReLU())
			d_bio = d_bio * 2
		self.bio_extractor = nn.Sequential(*bio_layers)

		self.gamma = nn.Linear(d_bio, d_model)
		self.beta = nn.Linear(d_bio, d_model)

	def forward(self, x):
		bio = self.bio_extractor(x)
		gamma = self.gamma(bio)
		beta = self.beta(bio)
		return gamma, beta