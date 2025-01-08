import os
import pandas as pd
import json
from aste_data_utils import Data as aste_data
from DataOrder import OrderedDataset

FEWREL_LABELS = [
    "P1001",
    "P101",
    "P102",
    "P105",
    "P106",
    "P118",
    "P123",
    "P127",
    "P1303",
    "P131",
    "P1344",
    "P1346",
    "P135",
    "P136",
    "P137",
    "P140",
    "P1408",
    "P1411",
    "P1435",
    "P150",
    "P156",
    "P159",
    "P17",
    "P175",
    "P176",
    "P178",
    "P1877",
    "P1923",
    "P22",
    "P241",
    "P264",
    "P27",
    "P276",
    "P306",
    "P31",
    "P3373",
    "P3450",
    "P355",
    "P39",
    "P400",
    "P403",
    "P407",
    "P449",
    "P4552",
    "P460",
    "P466",
    "P495",
    "P527",
    "P551",
    "P57",
    "P58",
    "P6",
    "P674",
    "P706",
    "P710",
    "P740",
    "P750",
    "P800",
    "P84",
    "P86",
    "P931",
    "P937",
    "P974",
    "P991",
]

def load_train_data(datadir, dataset="sst2", synthetic=False):
    '''
    Returns the training dataframe train_df and number of labels as an int
    '''
    fname = "syn-train" if synthetic else "train"

    if dataset == "sst2":
        path = os.path.join(datadir,"SST2")
        train_df = pd.read_csv(os.path.join(path, fname+".tsv"),sep='\t',header=0)
        train_df.columns = ['sentence', 'label']
        return train_df, 2 
    elif dataset == "fewrel":
        path = os.path.join(datadir,"FewRel")
        train_df = load_fewrel_data(os.path.join(path, fname+".json"))
        return train_df, len(FEWREL_LABELS)
    elif dataset == "aste":
        path = os.path.join(datadir,"ASTE/PGDG1.txt")
        train_df = aste_data.load_from_full_path(path)
        return train_df
    else:
        raise ValueError('Invalid dataset name passed!')

def load_test_data(datadir, dataset="sst2"):
    if dataset == "sst2":
        path = os.path.join(datadir,"SST2")
        test_df = pd.read_csv(os.path.join(path,"test.tsv"),sep='\t',header=0)
        test_df.columns = ['label','sentence']
        test_df = test_df[['sentence', 'label']]
        return test_df
    elif dataset == "fewrel":
        path = os.path.join(datadir,"FewRel")
        test_df = load_fewrel_data(os.path.join(path,"test.json"))
        return test_df
    else:
        raise ValueError('Invalid dataset name passed!')

# ------ ASTE Data Functions ------ #


# ------ FewRel Data Functions ------- #
def linearize_input(text, head, tail):
    return f"Head Entity : {head} , Tail Entity : {tail} , Context : {text}"


def read_sample_dict(sample):
    tokens = sample["tokens"]
    head = " ".join([tokens[i] for i in sample["h"][2][0]])
    tail = " ".join([tokens[i] for i in sample["t"][2][0]])
    return " ".join(tokens), head, tail


def load_fewrel_data(path):
    pairs = []
    with open(path) as f:
        raw = json.load(f)
        for label, lst in raw.items():
            y = FEWREL_LABELS.index(label)
            for sample in lst:
                text, head, tail = read_sample_dict(sample)
                x = linearize_input(text, head, tail)
                pairs.append((x, y))

    df = pd.DataFrame(pairs)
    df.columns = ["sentence", "label"]
    df = df.sample(frac=1)  # Shuffle
    print(dict(path=path, data=df.shape, unique_labels=len(set(df["label"].tolist()))))
    return df