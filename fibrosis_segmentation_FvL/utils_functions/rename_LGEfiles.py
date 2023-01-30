import os
from tqdm import tqdm

splits = ['train', 'validation', 'test']
path = '/home/flieshout/deep_risk_models/data/AUMC_classification_data'

for split in tqdm(splits):
    folder_path = os.path.join(path, split)
    obj_names = os.listdir(folder_path)
    for fn in obj_names:
        pat_id = fn.split('_')[0]
        new_fn = f'{pat_id}_LGE_PSIR.mha'
        os.rename(os.path.join(folder_path, fn), os.path.join(folder_path, new_fn))

