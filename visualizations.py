from visualization_utils import TruthData
import torch
from plotly.subplots import make_subplots
import argparse
import os


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def generate_plot(model, dataset, layer, color='label'):
	fig = TruthData.from_datasets([dataset], # datasets to use
		model_size=model,
		layer=layer,
		center=True,
		device=device).plot(
		dimensions=2, # 3 dimensions also supported
		dim_offset=0, # increase if you want to ignore the first few PCs
		color=color)
	return fig


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate plots from generated activations")
	parser.add_argument("--model", default="13B", help="Size of the model to use. Options are 7B or 30B")	
	parser.add_argument("--layers", nargs='+', help="Layers to save embeddings from")
	parser.add_argument("--datasets", nargs='+', help="Names of datasets, without .csv extension")
	parser.add_argument("--output_dir", default="plots", help="Directory to save activations to")
	parser.add_argument("--color", default="label", help="Dataset column to use for color labels")


	args = parser.parse_args()



	for layer in args.layers:
		savedir = f"{args.output_dir}/{args.model}/{layer}"
		if not os.path.exists(savedir):
			os.makedirs(savedir)

		for ds in args.datasets:
			f = generate_plot(args.model, ds, layer, args.color)			
			f.write_image(f"{savedir}/{ds}.png")
			

	

	


