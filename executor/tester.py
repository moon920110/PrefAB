import os

import dtw
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

from network.prefab import Prefab


class RanknetTester:
    def __init__(self, dataset, testset, config, logger):
        self.config = config
        self.logger = logger
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        self.batch_size = config['test']['batch_size']

        self.meta_feature_size = dataset.get_meta_feature_size()
        self.bio_features_size = dataset.bio_features_size
        self.test_dataset = testset

        # torch.set_float32_matmul_precision('high')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tsne = TSNE(n_components=2, random_state=config['test']['seed'])

    def test(self):
        writer = SummaryWriter(
            log_dir=os.path.join(self.config['test']['log_dir'],
                                 f"{self.config['test']['exp']}"
                                 )
        )
        label_path = os.path.join(self.config['test']['log_dir'], f"{self.config['test']['exp']}", 'metadata.tsv')

        model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)
        model.load_state_dict(
            torch.load(
                os.path.join(self.config['test']['save_dir'],
                             f'ranknet_{self.config["train"]["exp"]}_best.pth')
            )
        )
        model.to(self.device)
        model.eval()

        self.logger.info(f'Model loaded from {self.config["test"]["save_dir"]}/{self.config["train"]["exp"]}_best.pth')
        self.logger.info(f'Metadata will be saved at {label_path}')
        with torch.no_grad():
            # model save if validation accuracy is the best
            indices = self.test_dataset.sample_player_data(self.config['test']['sample_size'])
            tsne = TSNE(n_components=2, random_state=self.config['test']['seed'])
            metadata = []
            embeddings = []

            for i, idx in enumerate(indices):
                start_idx = self.test_dataset.player_idx[idx]
                end_idx = self.test_dataset.player_idx[idx + 1]

                outputs = []
                labels = []
                imgs = []
                features = []
                bios = []
                batch_idx = 0
                for data_idx in tqdm(range(start_idx, end_idx), desc=f'Reconstructing Graphs {i}/{len(indices)}'):
                    img, feature, bio, y = self.test_dataset[data_idx]
                    imgs.append(img)
                    features.append(feature)
                    bios.append(bio)
                    labels.append(y)

                    metadata.append([bio[:, 0], bio[:, 1], bio[:, 2],bio[:, 3], bio[:, 4], bio[:, 5], bio[:, 6], bio[:, 7]])

                    if len(imgs) == self.batch_size or data_idx == end_idx - 1:
                        batch_idx += 1
                        imgs = torch.stack(imgs).to(self.device)
                        features = torch.stack(features).to(self.device)
                        bios = torch.stack(bios).to(self.device)

                        o, _, z = model(imgs, features, bios, test=True)
                        o = o.cpu().detach().numpy()
                        outputs.extend(o)

                        flat_z = z.view(z.size(0), -1).cpu().detach().numpy()
                        embeddings.append(flat_z)

                        imgs = []
                        features = []
                        bios = []

                # normalize output to 0~1
                outputs = np.array(outputs).squeeze().squeeze()
                outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
                # dtw_distances.append(dtw.dtw(outputs, np.array(labels)).distance)
                # tsne_embeddings = tsne.fit_transform(embeddings)

                for ii, (o, y) in enumerate(zip(outputs, labels)):
                    if writer:
                        writer.add_scalars(f'test/player_{idx}',
                                           {'predict': o,
                                            'arousal': y,
                                            },
                                           ii)

                # dtw_distances = np.array(dtw_distances)
                # if writer:
                #     writer.add_scalar(f'test/dtw_mean', dtw_distances.mean(), epc)
                #     writer.add_scalar(f'test/dtw_std', dtw_distances.std(), epc)
            embeddings = np.vstack(embeddings)
            writer.add_embedding(
                torch.tensor(embeddings),
                metadata_header=['Age', 'Gender', 'Frequency', 'Gamer', 'PC', 'Mobile', 'Console', 'Genre'],
                metadata=metadata,
                # tag=f't-SNE Embeddings {i}',
            )
        if writer is not None:
            writer.close()

    def _metric(self, y_pred, y_true, cutpoints):
        _y_pred = y_pred.cpu().detach().numpy()
        _y_pred = np.where(_y_pred < cutpoints[0], 0, np.where(_y_pred < cutpoints[1], 1, 2))

        acc = accuracy_score(y_true.cpu().detach().numpy(), _y_pred)
        cm = confusion_matrix(y_true.cpu().detach().numpy(), _y_pred, labels=[0, 1, 2])

        return acc, cm

