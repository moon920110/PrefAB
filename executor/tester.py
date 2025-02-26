import os

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from network.prefab import Prefab
from utils.utils import normalize


class RanknetTester:
    def __init__(self, testset, bio_feature_size, config, logger):
        self.config = config
        self.logger = logger
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        self.batch_size = config['test']['batch_size']

        self.meta_feature_size = testset.get_meta_feature_size()
        self.bio_features_size = bio_feature_size
        self.test_dataset = testset

        # torch.set_float32_matmul_precision('high')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def inference(self):
        pass

    def test(self, writer=None, model_path=None):
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

                # normalize output to 0~1
                outputs = np.array(outputs).squeeze().squeeze()
                outputs = normalize(outputs)

                for ii, (o, y) in enumerate(zip(outputs, labels)):
                    if test_writer:
                        test_writer.add_scalars(f'test/player_{idx}',
                                           {'predict': o,
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
