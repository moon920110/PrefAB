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
from utils.utils import normalize


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

    def test(self, writer=None, model_path=None, cutpoints=None):
        if writer is None:
            test_writer = SummaryWriter(
                log_dir=os.path.join(self.config['test']['log_dir'],
                                     f"{self.config['test']['new_exp']}"
                                     )
            )
            label_path = os.path.join(self.config['test']['log_dir'], f"{self.config['test']['new_exp']}", 'metadata.tsv')
            self.logger.info(f'Metadata will be saved at {label_path}')
        else:
            test_writer = writer

        model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)

        if model_path is None:
            model_path = os.path.join(self.config['test']['save_dir'], f'ranknet_{self.config["test"]["exp"]}_best.pth')

        if cutpoints is None:
            cutpoints = self.config['test']['cutpoints']

        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

        self.logger.info(f'Model loaded from {model_path}')
        with torch.no_grad():
            # model save if validation accuracy is the best
            indices = self.test_dataset.sample_player_data(self.config['test']['sample_size'])
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

                for data_idx in tqdm(range(start_idx, end_idx), desc=f'Reconstructing Graphs {i}/{len(indices)}'):
                    img, feature, bio, y, cluster = self.test_dataset[data_idx]
                    imgs.append(img)
                    features.append(feature)
                    bios.append(bio)
                    labels.append(y)

                    metadata.append([f'{y:.2f}', cluster])

                    if len(imgs) == self.batch_size or data_idx == end_idx - 1:
                        imgs = torch.stack(imgs).to(self.device)
                        features = torch.stack(features).to(self.device)
                        bios = torch.stack(bios).to(self.device)

                        o, _, _, z = model(imgs, features, bios, test=True)
                        o = o.cpu().detach().numpy()
                        outputs.extend(o)

                        flat_z = z.view(z.size(0), -1).cpu().detach().numpy()
                        embeddings.append(flat_z)

                        imgs = []
                        features = []
                        bios = []

                outputs = np.array(outputs).squeeze().squeeze()
                relative_outputs = []
                stride = self.config['train']['window_stride']
                rel_graph = 0
                for o_idx in range(len(outputs) - stride):
                    if o_idx == 0:
                        rel_val = 0
                    else:
                        rel_val = outputs[o_idx + stride] - outputs[o_idx]

                    if rel_val < cutpoints[0]:
                        rel_graph += -1
                    elif cutpoints[0] < rel_val < cutpoints[1]:
                        rel_graph += 0
                    else:
                        rel_graph += 1

                    relative_outputs.append(rel_graph)
                relative_outputs.extend([0, 0, 0, 0])
                relative_outputs = np.array(relative_outputs)

                # normalize output to 0~1
                outputs = normalize(outputs)
                relative_outputs = normalize(relative_outputs)

                for ii, (o, y, ro) in enumerate(zip(outputs, labels, relative_outputs)):
                    if test_writer:
                        test_writer.add_scalars(f'test/player_{idx}',
                                           {'predict': o,
                                            'predict_rel': ro,
                                            'arousal': y,
                                            },
                                           ii)

            embeddings = np.vstack(embeddings)
            test_writer.add_embedding(
                torch.tensor(embeddings),
                metadata,
                metadata_header=['Arousal', 'Cluster'],
            )

        if writer is None:
            test_writer.close()
