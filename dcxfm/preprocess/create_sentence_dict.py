'''
This script creates a dictionary of all sentences we will used and their labels by ChexBert. 
'''
from argparse import ArgumentParser
import os
import ipdb
import json
import re
import nltk
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


# The mapping of dataset name to raw report list.
dataset_dir = "/disk1/fywang/CXR_dataset"
REPORT_DICT = {
    "mimic-cxr": os.path.join(dataset_dir, "mimic_data/2.0.0/master.csv"),
    "iu-xray": os.path.join(dataset_dir, "chest-xrays-indiana-university/indiana_reports.csv"),
    "padchest": os.path.join(dataset_dir, "BIMCV-PadChest-FULL/PADCHEST_chest_x_ray_translated.csv"),
}

parser = ArgumentParser()
parser.add_argument('--dataset_list', nargs="+", 
                    default=["mimic-cxr", "iu-xray", "padchest"])
args = parser.parse_args()


def _split_report_into_segment(report):
    '''clean up raw reports into sentences
    This preprocessing function is borrowed from MedCLIP.
    '''
    if pd.isnull(report):
        return []
    else:
        report = report.replace('\n',' ')
        # splitter = re.compile("[0-9]+\.")
        splitter = re.compile("[0-9]+\.+[^0-9]")
        report = splitter.split(report)
        reports = [point.split(". ") for point in report]
        # reports = [point.split(".") for point in report]
        reports = [sent for point in reports for sent in point]
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            # tokenizer = RegexpTokenizer(r"\w+")
            # tokens = tokenizer.tokenize(sent.lower())

            tokens = tokens = nltk.wordpunct_tokenize(sent.lower())
            
            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)
            if len(included_tokens) > 4: # only include relative long sentences
                study_sent.append(" ".join(included_tokens))
        return study_sent


def main():
    all_sentences = []

    for dataset in args.dataset_list:
        print("Processing dataset: {}".format(dataset))
        if dataset == "padchest":
            dataset_df = pd.read_csv(REPORT_DICT[dataset], low_memory=False)
            dataset_df = dataset_df.dropna(subset=["Report_Eng"])
            dataset_df["report"] = dataset_df["Report_Eng"].apply(_split_report_into_segment)
        elif dataset == "mimic-cxr":
            dataset_df = pd.read_csv(REPORT_DICT[dataset], low_memory=False)
            dataset_df["report"] = dataset_df["impression"] + " " + dataset_df["findings"]
            dataset_df = dataset_df.dropna(subset=["report"])
            dataset_df = dataset_df.drop_duplicates(subset=["report"])
            dataset_df["report"] = dataset_df["report"].apply(_split_report_into_segment)
        elif dataset == "iu-xray":
            dataset_df = pd.read_csv(REPORT_DICT[dataset], low_memory=False)
            dataset_df = dataset_df.dropna(subset=["findings"])
            dataset_df = dataset_df.drop_duplicates(subset=["findings"])
            dataset_df["report"] = dataset_df["findings"].apply(_split_report_into_segment)

        # all_sentences.extend(dataset_df["report"].tolist())
        for row in dataset_df.itertuples():
            # we treat each sentence as the basic unit.
            all_sentences.extend(row.report)
    
    all_sentences = list(set(all_sentences))

    sent_df = pd.DataFrame(all_sentences, columns=["Report Impression"])
    sent_df = sent_df.drop_duplicates(subset=["Report Impression"])
    # has_comparison = sent_df["Report Impression"].str.contains("compare", case=False, regex=False)
    # sent_df = sent_df[~has_comparison]
    sent_df.reset_index(drop=True, inplace=True)
    dataset_str = "_".join(args.dataset_list)
    sent_df.to_csv(os.path.join(dataset_dir, f"mask/reports/sentence_dict_{dataset_str}.csv"), index=False)


'''
Next, we use chexbert to get sentence labels.
### CheXBert usage
Commands:
```
conda create -n chexbert python=3.7
conda activate chexbert

python label.py -d /home/r15user2/Documents/CXR_dataset/mask/reports/sentence_dict_mimic-cxr_iu-xray_padchest.csv -o /home/r15user2/Documents/CXR_dataset/mask/reports -c /home/r15user2/Documents/CXRSeg/pretrained/chexbert/chexbert.pth
```

Torch installation:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
'''
if __name__ == '__main__':
    main()