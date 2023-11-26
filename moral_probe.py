import torch as t
from utils import DataManager
import random
#import matplotlib.pyplot as plt
import random
from probes import LRProbe, MMProbe, CCSProbe

# hyperparameters
model_size = '13B'
layer = 12 # layer from which to extract activations
split = 0.8
seed = random.randint(0, 100000)
device = 'cuda:0' if t.cuda.is_available() else 'cpu'


ProbeClasses = [MMProbe, LRProbe]


accs = {}


dm = DataManager()
dm.add_dataset("moral", model_size, layer, split=split, seed=seed, device=device)

train_acts, train_labels = dm.get('train')

for ProbeClass in ProbeClasses:
	probe = ProbeClass.from_data(train_acts, train_labels, device=device)
	val_acts, val_labels = dm.get('val')
	scores = probe.pred(val_acts) == val_labels
	accs[str(ProbeClass)] = scores.float().mean().item()


print(accs)

