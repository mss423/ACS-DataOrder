import pandas as pd
import os
from acs import *
import kmeans_sample


def get_k_sample(K, dataset="SST2", synthetic=False, method="random"):
	data_path  = "" + dataset
	syn_pre    = "syn" if synthetic else "human"
	sub_file   = method + "_" + syn_pre + f"_K{K}.tsv"

	if method == "random":
		# Load data if available
		try:
			sub_df  = pd.read_csv(os.path.join(data_path, sub_file), sep='\t')
			print("Subsample exists, loading ...")

		# Else run sampling on full datafile
		except:
			print("Running downsamping ...")
			data_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t')
			sub_df  = data_df.sample(n=K, random_state=42)
			sub_df.to_csv(os.path.join(data_path,sub_file), sep='\t')


	elif method == "kmeans":
		try:
			sub_df  = pd.read_csv(os.path.join(data_path, sub_file), sep='\t')
			print("Subsample exists, loading ...")
		except:
			print("Running downsamping ...")
			data_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t')
			sub_df  = kmeans_sample(data_df, K)
			sub_df.to_csv(os.path.join(data_path,sub_file), sep='\t')
			

	elif method == "acs":
		try:
			sub_df  = pd.read_csv(os.path.join(data_path, sub_file), sep='\t')
			print("Subsample exists, loading ...")
		except:
			print("Running downsamping ...")
			data_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t')
			sub_df  = acs_sample(data_df, K)
			sub_df.to_csv(os.path.join(data_path,sub_file), sep='\t')


	elif method == "judge":
		try:
			sub_df  = pd.read_csv(os.path.join(data_path, sub_file), sep='\t')
			print("Subsample exists, loading ...")
		except:
			raise ValueError('Have not implemented yet ...')

	else:
		raise ValueError('Invalid downsampling method passed!')


	# Return dataframe
	return sub_df
