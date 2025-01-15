import os
import pandas as pd

from utils.stats import get_dtw_cluster


class BioReader:
	def __init__(self, again, config=None, logger=None):
		self.again = again
		self.config = config
		self.logger = logger

		self.data_path = config['data']['path']
		self.bio = pd.read_csv(os.path.join(self.data_path, 'raw_data', 'biographical_data_with_genre.csv'),
		                       encoding='utf-8',
		                       low_memory=False)
		self.bio_size = None

		self._set_cluster()
		self._calc_bio_size()
		self.bio = self.bio.drop(columns=['ParticipantID', 'Metacritic Code', 'Genre'])

		# normalize the age between 0 and 80
		self.bio['Age'] = self.bio['Age'] / 80

	def _calc_bio_size(self):
		self.bio['genre_idx'] = pd.factorize(self.bio['Genre'])[0]
		gender_size = self.bio['Gender'].unique().size
		play_freq_size = self.bio['Play Frequency'].unique().size
		gamer_size = self.bio['Gamer'].unique().size
		genre_size = self.bio['genre_idx'].unique().size
		self.bio_size = {'gender': gender_size, 'play_freq': play_freq_size, 'gamer': gamer_size, 'genre': genre_size}

	def _set_cluster(self):
		domain = self.config['train']['domain']
		scope = self.config['train']['genre'] if domain == 'genre' else self.config['train']['game']

		if domain == 'game':
			again = self.again.game_info_by_name(scope)
			again['game_idx'] = 0
		elif domain == 'genre':
			again = self.again.game_info_by_genre(scope)
			games = again['game'].unique()
			for game in games:
				again.loc[again['game'] == game, 'game_idx'] = self.config['game_numbering'][scope][game] / self.config['game_numbering']['game_cnt_per_genre']
		else:
			again = self.again

		if self.config['clustering']['load_cache']:
			clusters = pd.read_csv(os.path.join(self.data_path, 'cluster', f'cluster_{scope}.csv'))
		else:
			clusters = get_dtw_cluster(again, self.config)

			if self.config['clustering']['caching']:
				if not os.path.exists(os.path.join(self.data_path, 'cluster')):
					os.mkdir(os.path.join(self.data_path, 'cluster'))
				clusters = pd.DataFrame(clusters.items(), columns=['session_id', 'cluster'])
				clusters.to_csv(os.path.join(self.data_path, 'cluster', f'cluster_{scope}.csv'), index=False)

		for _, row in clusters.iterrows():
			p = again[again['session_id'] == row['session_id']]['player_id'].unique()[0]
			bio_idx = self.bio[self.bio['ParticipantID'] == p].index[0]
			self.bio.loc[bio_idx, 'cluster'] = row['cluster']

		# remove rows where cluster is NaN
		self.bio = self.bio.dropna(subset=['cluster'])