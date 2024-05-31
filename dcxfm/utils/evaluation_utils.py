import warnings
import torch
import numpy as np
import pandas as pd
from termcolor import colored
import random
from dcxfm.utils.prompts import generate_class_prompts
random.seed(42)
warnings.filterwarnings("ignore")


@torch.no_grad()
def get_class_texts(tokenizer, class_names, prompt_style="xplainer", num_templates=4, use_negative=False):
    '''
    Generate prompts for zero-shot segmentation.
    '''
    pos_class_prompts = generate_class_prompts(
        class_names, mode="pos", prompt_style=prompt_style)
    tokenized_pos_text = dict()
    for pathology in class_names:
        pos_texts = pos_class_prompts[pathology]
        if len(pos_texts) > num_templates:
            pos_texts = random.sample(pos_texts, num_templates)
        # FIXME: don't change this, don't know why ...
        pos_texts = tokenizer(pos_texts, return_tensors="pt",
                              padding="max_length", max_length=77)
        pos_texts["input_ids"] = pos_texts["input_ids"].cuda()
        pos_texts["attention_mask"] = pos_texts["attention_mask"].cuda()
        tokenized_pos_text[pathology] = pos_texts
    if use_negative:
        neg_class_prompts = generate_class_prompts(
            class_names, mode="neg", prompt_style=prompt_style)
        tokenized_neg_text = dict()
        for pathology in class_names:
            neg_texts = neg_class_prompts[pathology]
            if len(neg_texts) > num_templates:
                neg_texts = random.sample(neg_texts, num_templates)
            neg_texts = tokenizer(
                neg_texts, return_tensors="pt", padding="max_length", max_length=77)
            neg_texts["input_ids"] = neg_texts["input_ids"].cuda()
            neg_texts["attention_mask"] = neg_texts["attention_mask"].cuda()
            tokenized_neg_text[pathology] = neg_texts

        return tokenized_pos_text, tokenized_neg_text
    else:
        return tokenized_pos_text


def color_print(text, color="red"):
    return print(colored(text, color))


def bootstrap_metric(df, num_replicates, class_names):
    """Create dataframe of bootstrap samples."""
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df[class_names].columns:
            performance = np.nanmean(df_replicate[task].values)
            replicate_performances[task] = performance
        return replicate_performances

    all_performances = []
    for _ in range(num_replicates):
        replicate_performances = single_replicate_performances()
        all_performances.append(replicate_performances)

    df_performances = pd.DataFrame.from_records(all_performances)
    return df_performances


def compute_cis(series, confidence_level):
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(), 3)
    return lower, mean, upper


def create_ci_record(perfs, task, metric_name):
    lower, mean, upper = compute_cis(perfs, confidence_level=0.05)
    record = {"name": task,
              f"{metric_name}_lower": lower,
              f"{metric_name}_mean": mean,
              f"{metric_name}_upper": upper}
    return record
