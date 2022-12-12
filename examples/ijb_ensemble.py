import os
import argparse
import yaml
import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import sys
sys.path.append(os.getcwd())

# device = torch.devide('cuda')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='/home/k64t/face_recognition/configs/ensemble/equal_1.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--p", type=str, help="the saved file path")
    return parser.parse_args()

def plot_roc_and_calculate_tpr(scores, names=None, label=None, weights=None, output=None):
    print(">>>> plot roc and calculate tpr...")
    # for id, score in enumerate(scores):
    score_dict = {}
    if type(scores) is list:
        scores_list = []
        for index, score_model in list(enumerate(scores)):
            print(score_model)
            aa = np.load(score_model)
            score = aa.get("scores", [])
            scores_list.append(score)
            # label = aa["label"] if label is None and "label" in aa else label
        score = np.average(np.array(scores_list), axis=0, weights=weights)
        label = pd.read_csv(label, sep=" ", header=None).values[:, 2]
        score_name = aa.get("names", [])
        for ss, nn in zip(score, score_name):
            score_dict[nn] = ss    
    else:
        name = None if names is None else names[id]
        if isinstance(scores, str) and scores.endswith(".npz"):
            print('l')
            aa = np.load(scores)
            score = aa.get("scores", [])
            # label = aa["label"] if label is None and "label" in aa else label
            label = pd.read_csv(label, sep=" ", header=None).values[:, 2]
            score_name = aa.get("names", [])
            for ss, nn in zip(score, score_name):
                score_dict[nn] = ss

    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_dict, tpr_dict, roc_auc_dict, tpr_result = {}, {}, {}, {}
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # select largest tpr at same fpr
        tpr_result[name] = [tpr[np.argmin(abs(fpr - ii))] for ii in x_labels]
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df['AUC'] = pd.Series(roc_auc_dict)
    tpr_result_df.columns.name = "Methods"
    print(tpr_result_df.to_markdown())
    # print(tpr_result_df)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(fpr_dict[name], tpr_dict[name], lw=1, label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100))
        title = "ROC on IJB" + name.split("IJB")[-1][0] if "IJB" in name else "ROC on IJB"

        plt.xlim([10 ** -6, 0.1])
        plt.xscale("log")
        plt.xticks(x_labels)
        plt.xlabel("False Positive Rate")
        plt.ylim([0.3, 1.0])
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title(f'{title} {output}')
        plt.legend(loc="lower right", fontsize='x-small')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'/home/k64t/face_recognition/IJB_result/{output}.png')
    except:
        print("matplotlib plot failed")
        fig = None

    return tpr_result_df, fig

if __name__ == "__main__":
    args = get_args()

    with open(args.c, 'r') as file:
        print(args.c)
        config = yaml.load(file, Loader=yaml.Loader)

    scores = []
    weights = []
    for model_path in config['weight_path']:
        scores.append(config['weight_path'][model_path])
        weights.append(config["ensemble_weights"][model_path])

    output = args.c.replace('/home/k64t/face_recognition/configs/ensemble/', '')

    plot_roc_and_calculate_tpr(scores, names=None, label='/home/k64t/face_recognition/data/ijb/IJBB/IJBB_meta/ijbb_template_pair_label.txt', weights=weights, output=output)
    # fig.save('/home/k64t/face_recognition/IJB_result/test.png')

