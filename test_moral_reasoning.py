import csv
import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import random

os.chdir("/w/339/bkuwahara/mat1510/MAT1510-project")
llama_path = "/w/339/bkuwahara/llama_model/13b"

config = configparser.ConfigParser()
config.read('config.ini')
LLAMA_DIRECTORY = config['LLaMA']['weights_directory']

if not os.path.exists(LLAMA_DIRECTORY):
	raise Exception("Make sure you've set the path to your LLaMA weights in config.ini")


def load_llama(model_size):
	llama_path = os.path.join(LLAMA_DIRECTORY, config['LLaMA'][f'{model_size}_subdir'])
	tokenizer = LlamaTokenizer.from_pretrained(llama_path)
	model = LlamaForCausalLM.from_pretrained(llama_path, device_map="auto", offload_folder="offload")
	# set tokenizer to use bos token
	tokenizer.bos_token = '<s>'
	#if model_size == '13B' and device != 'cpu':
	#   model = model.half()
	#model.to(device)
	return tokenizer, model

def load_statements(dataset_name):
	"""
	Load statements from csv file, return list of strings.
	"""
	dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
	statements = dataset['statement'].tolist()
	answers = dataset['label'].tolist()
	return statements, answers




if __name__ == "__main__":
	"""
	read statements from dataset, record activations in given layers, and s>
	"""
	parser = argparse.ArgumentParser(description="Test LLaMa-2's ability to perform moral reasoning on MMLU moral reasoning statements.")
	parser.add_argument("--model", default="13B",
	                help="Size of the model to use. Options are 7B, 13B, 70B")
	parser.add_argument("--k", default="0", type=int,
	                help="Number of examples to give the model before asking for its response")

	args = parser.parse_args()

	k = args.k
	size = args.model
	tokenizer, model = load_llama(size)
		
	
	data = pd.read_csv("./datasets/moral.csv")

	
	model_answers = []
	for i in range(1,len(data)):	
		prompt = "In the following scenario(s), does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Respond with True or False.\n "
		q = data.loc[i]
		rest = data.loc[~data.index.isin([i])]
		ex = rest.sample(n=k)
		for j in ex.index:
			statement, label = ex["statement"][j], ex["label"][j]
			correct_answer = False if label == 1 else True
			prompt += "Scenario: {}\nAnswer: {}\n".format(statement, correct_answer)
		
		prompt += "Scenario: {}\nAnswer: ".format(q["statement"])
	
		input = tokenizer(prompt, return_tensors="pt").to("cuda")
		generate_ids = model.generate(**input, max_new_tokens=3, temperature=0)
		out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

		model_answer = out.rsplit("Answer: ", maxsplit=1)[-1].split()[0]
		to_label = 1 if model_answer == False else 0
		model_answers.append(to_label)			
	

	data.insert(2, "model_answer", model_answers)
	data.to_csv("./datasets/moral_with_outputs.csv")
