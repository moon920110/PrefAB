# Extract bio to clustere
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn


class BioNet(nn.Module):
	def __init__(self, config, bio_feature_size):
		super(BioNet, self).__init__()
		self.config = config

		self.genre_size = bio_feature_size['genre']
		self.gender_size = bio_feature_size['gender']
		self.play_freq_size = bio_feature_size['play_freq']
		self.gamer_size = bio_feature_size['gamer']
		bio_embedding_dim = config['train']['bio_embedding_dim']

		# self.age_embedding = nn.Linear(1, bio_embedding_dim)
		self.genre_embedding = nn.Embedding(self.genre_size, bio_embedding_dim)
		bio_total_size = (
			1 +
			self.gender_size +
			self.play_freq_size +
			self.gamer_size +
			3 +
			bio_embedding_dim
		)

		self.bio_extractor = nn.Sequential(nn.Linear(bio_total_size, 128),
		                                   nn.LayerNorm(128),
		                                   nn.LeakyReLU(),
		                                   nn.Linear(128, 128),
		                                   nn.LayerNorm(128),
		                                   nn.LeakyReLU(),
		                                   nn.Dropout(self.config['train']['dropout']),
		                                   nn.Linear(128, 64),
		                                   nn.LayerNorm(64),
		                                   nn.LeakyReLU(),
		                                   nn.Dropout(self.config['train']['dropout']),
		                                   nn.Linear(64, 32),
		                                   nn.LayerNorm(32),
		                                   nn.LeakyReLU(),
		                                   nn.Dropout(self.config['train']['dropout']),
		                                   nn.Linear(32, config['clustering']['n_clusters'])
		                                   )
		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# if m is self.bio_extractor:
				# 	init.xavier_normal_(m.weight)
				# else:
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1.0)
				init.constant_(m.bias, 0.0)

	def forward(self, bio):
		bio = bio.squeeze(1)

		age = bio[:, 0].float()
		gender = bio[:, 1].long()
		freq = bio[:, 2].long()
		gamer = bio[:, 3].long()
		platform = bio[:, 4:7].float()
		genre = bio[:, 7].long()

		# age_emb = self.age_embedding(age.unsqueeze(1).float())
		gender_onehot = F.one_hot(gender, num_classes=self.gender_size).float()
		freq_onehot = F.one_hot(freq, num_classes=self.play_freq_size).float()
		gamer_onehot = F.one_hot(gamer, num_classes=self.gamer_size).float()
		genre_emb = self.genre_embedding(genre)

		# bio_feature = torch.cat([age_emb, gender_onehot, freq_onehot, platform, gamer_onehot, genre_emb], dim=1)
		bio_feature = torch.cat([age.unsqueeze(1), gender_onehot, freq_onehot, platform, gamer_onehot, genre_emb], dim=1)
		o = self.bio_extractor(bio_feature)

		return o



