from datasets import BebopPitchDurDataset
from dataloaders import SplitDataLoader
import pdb

ds = BebopPitchDurDataset()
train_loader, valid_loader = SplitDataLoader(ds).split()
pdb.set_trace()
it = iter(train_loader)
