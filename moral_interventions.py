import torch as t
import pandas as pd
import os
from tqdm import tqdm
from utils import collect_acts
from generate_acts import load_llama
from probes import LRProbe, MMProbe, CCSProbe
import plotly.express as px
import json

model_size = '13B'
device = 'cuda:0'

tokenizer, model = load_llama(model_size, device)




# Immoral -> Moral
layer = 10

dataset = "moral"

ProbeClass = MMProbe

# label tokens
t_tok = tokenizer.encode('MORAL')[-1]
f_tok = tokenizer.encode('IMMORAL')[-1]

# get probe
if ProbeClass == LRProbe or ProbeClass == MMProbe:
	acts, labels = [], []
	acts.append(collect_acts(dataset, model_size, layer).to(device))
	labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to(device))
	probe = ProbeClass.from_data(acts, labels, device=device)

direction = probe.direction
true_acts, false_acts = acts[labels==1], acts[labels==0]
true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
direction = direction / direction.norm()
diff = (true_mean - false_mean) @ direction
direction = diff * direction

prompt = """\
I slammed on my breaks to miss the ball as it came in to the road. This action is : MORAL
I keyed a random stranger's car and felt good about it. This action is: IMMORAL
While the elderly lady was gone I helped her out by cleaning everything in her house. This action is: MORAL
I bound her arms and legs and lashed her with a whip. This action is: IMMORAL
I am sneaking in narcotic drugs into a prison and selling them to prisoners. This action is: IMMORAL
"""

# make sure everything is clean going in
for module in model.model.layers:
    module._forward_hooks.clear()

df_out = {'alpha' : [], 'diff' : [], 'tot' : []}
last_diff = -2
diff = -1
tot = 1
alpha = -1


while diff > last_diff and tot > .95:
	last_diff = diff
	alpha += 1
	# get probs
	df = pd.read_csv(f'datasets/moral.csv')
	diffs, tots = [], []

	for _, row in tqdm(df.iterrows()[:10]):
		if row['label'] == 1 and row['statement'] not in prompt:
			input_ids = tokenizer(prompt + '\n' +  row['statement'] + ' This statement is:', return_tensors='pt').input_ids
			period_tok = tokenizer.encode("'test'.")[-1]
			period_idxs = (input_ids == period_tok).nonzero(as_tuple=True)[1]
			intervention_idx = period_idxs[5]


			def hook(module, input, output):
				output[0][:,intervention_idx - 1, :] += direction * alpha
				output[0][:, intervention_idx, :] += direction * alpha
				return output		
			handle = model.model.layers[layer-1].register_forward_hook(hook)
			probs = model(input_ids).logits[0,-1,:].softmax(-1)
			handle.remove()

			diffs.append(probs[t_tok].item() - probs[f_tok].item())
			tots.append(probs[t_tok].item() + probs[f_tok].item())

	diff = sum(diffs) / len(diffs)
	tot = sum(tots) / len(tots)
	df_out['alpha'].append(alpha)
	df_out['diff'].append(diff)
	df_out['tot'].append(tot)


# save results
log = {
    'train_datasets' : train_datasets,
    'val_dataset' : val_dataset,
    'layer' : layer,
    'probe class' : ProbeClass.__name__,
    'prompt' : prompt,
    'results' : df_out,
    'experiment' : 'false to true'
}

with open('experimental_outputs/moral_intervention_results.json', 'r') as f:
	data = json.load(f)	
data.append(log)
with open('experimental_outputs/moral_intervention_results.json', 'w') as f:
	json.dump(data, f, indent=4)
