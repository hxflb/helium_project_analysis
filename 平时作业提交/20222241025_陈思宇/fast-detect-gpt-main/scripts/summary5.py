# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
import re
import pandas as pd
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from anti_detect import rewrite_text
# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text
    #print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    #print('')
    file_path = "data/arxiv/arxiv_data.csv"
    count = 0
    correct = 0
    error = 0
    processed_correct = 0
    processed_error = 0

    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
    texts = df["summaries"]
    texts = texts.head(1000)
    for text in texts:
        if isinstance(text, str): #and len(text) > 50:
            text = re.sub(r'[^a-zA-Z0-9.,!?/:\'\\\s]', '', text)
            if type(text) == str and len(text) > 10:
                print(text + "\n")
                count += 1
                tokenized = scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                                              return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = scoring_model(**tokenized).logits[:, :-1]
                    if args.reference_model_name == args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = reference_tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                                                        return_token_type_ids=False).to(args.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                        logits_ref = reference_model(**tokenized).logits[:, :-1]
                    crit = criterion_fn(logits_ref, logits_score, labels)
                # estimate the probability of machine generated text
                prob = prob_estimator.crit_to_prob(crit)
                #print(
                #    f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
                #print()
                if prob > 0.5:
                    predicted_class = 1
                else:
                    predicted_class = 0
                processed_text = rewrite_text(text)
                print("predicted_class:", predicted_class)
                print("processed:   " + processed_text + "\n")
                processed_tokenized = scoring_tokenizer(processed_text, truncation=True, return_tensors="pt", padding=True,
                                              return_token_type_ids=False).to(args.device)
                processed_labels = processed_tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    processed_logits_score = scoring_model(**processed_tokenized).logits[:, :-1]
                    if args.reference_model_name == args.scoring_model_name:
                        processed_logits_ref = processed_logits_score
                    else:
                        processed_tokenized = reference_tokenizer(processed_text, truncation=True, return_tensors="pt", padding=True,
                                                        return_token_type_ids=False).to(args.device)
                        assert torch.all(processed_tokenized.input_ids[:, 1:] == processed_labels), "Tokenizer is mismatch."
                        processed_logits_ref = reference_model(**processed_tokenized).logits[:, :-1]
                    processed_crit = criterion_fn(processed_logits_ref, processed_logits_score, processed_labels)
                # estimate the probability of machine generated text
                processed_prob = prob_estimator.crit_to_prob(processed_crit)
                # print(
                #    f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
                # print()
                if processed_prob > 0.5:
                    processed_class = 1
                else:
                    processed_class = 0
                print("processed_class:", processed_class)
                if predicted_class == 0:
                    correct += 1
                elif predicted_class == 1:
                    error += 1
                if processed_class == 0:
                    processed_correct += 1
                elif processed_class == 1:
                    processed_error += 1
    rate = (correct * 1.0) / (count * 1.0)
    processed_rate = (processed_correct * 1.0) / (count * 1.0)
    print("count:", count, "\ncorrect:", correct, "\nerror:", error, "\nrate:", rate * 100, "%")
    print("count:", count, "\nprocessed_correct:", processed_correct, "\nprocessed_error:", processed_error,
          "\nprocessed_rate:", processed_rate * 100, "%")
'''
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # evaluate text
'''









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="D:/Project/py/fast-detect-gpt-main/local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



