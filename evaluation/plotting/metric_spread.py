import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

STATS = ('wer', 'cer', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'jaro_winkler')
STATS_MAP = {
    'wer': "Word Error Rate", 
    'cer': "Character Error Rate", 
    'bertscore_precision': "BERTScore Precision", 
    'bertscore_recall': "BERTScore Recall", 
    'bertscore_f1': "BERTScore F1", 
    'jaro_winkler': "Jaro-Winkler Distance"
}
CANARY = {
    'canary': 1000,
}
W2V_MODELS = {
    'wav2vec2_large': 317,
    'wav2vec2_base': 95
}
MODELS = {
    'whisper_large': 1540,
    'whisper_medium': 764,
    'whisper_small': 242,
    'whisper_base': 72.6,
    'whisper_tiny': 37.8
}
MODELS_EN = {
    # 'whisper_large_en': 1540,
    'whisper_medium_en': 764,
    'whisper_small_en': 242,
    'whisper_base_en': 72.6,
    'whisper_tiny_en': 37.8
}


def process_df(df):
    stats = {s: None for s in STATS}

    count = len(df)
    for s in stats.keys():
        stdev = df[s].std()
        avg = df[s].mean()
        # stat: (mean, 95_conf_upper, 95_conf_lower, stdev)
        stats[s] = (avg, avg + (1.96*stdev/count), avg - (1.96*stdev/count), stdev)

    return stats


def get_stats_over_ag(f_path):
    df = pd.read_csv(f_path, sep='\t')
    df = df.drop(df.index[df['preprocessed_transcriptions'].isna()])
    df_by_ag = {ag.replace('/', '_').replace('-', '_').replace('.', '_').replace(' ', '_'): rows for ag, rows in df.groupby('accent_group')}
    
    res = {}
    for ag in df_by_ag.keys():
        res[ag] = process_df(df_by_ag[ag])

    # Return {accent_group: {stat: (mean, 95_conf_upper, 95_conf_lower, stdev)}}
    return res


def plot_model_group(model_group, out_dir, model_name, en=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # x = model_group.values()
    model_stats = {model: get_stats_over_ag(f'../results/{model}_all_results/sa_transcriptions.tsv') for model in model_group.keys()}
    accent_groups = list(model_stats[list(model_stats.keys())[0]].keys())

    print(model_stats.keys())

    for s in STATS:
        out_path = '/'.join([out_dir, f'{s + ("_en" if en else "")}.png'])

        x = []
        y_list = [[] for a in accent_groups]
        ci_u_list = [[] for a in accent_groups]
        ci_l_list = [[] for a in accent_groups]
        for m, size in model_group.items():
            x.append(size)
        for i, a in enumerate(accent_groups):
            for m in model_group.keys():
                y_list[i].append(model_stats[m][a][s][0])
                ci_u_list[i].append(model_stats[m][a][s][1])
                ci_l_list[i].append(model_stats[m][a][s][2])

    return model_stats

 
def latex_table(all_stats):

    stats = ['wer', 'cer', 'jaro_winkler']
    print('\hline')
    print(f'Model & WER & CER & JW-Dist \\\\')
    print('\hline')
    for model_stats in all_stats:
        accent_groups = list(model_stats[list(model_stats.keys())[0]].keys())
        models = list(model_stats.keys())
        for m in models:
            row = [m]
            for s in stats:
                stats_list = [model_stats[m][ag][s][0] for ag in accent_groups]
                row.append(str((max(stats_list) - min(stats_list)).__round__(5)))

            print(' & '.join(row).replace('_', '\\_') + ' \\\\')

        print('\hline')


def main():
    model_stats = []
    model_stats.append(plot_model_group(MODELS, 'whisper', 'Whisper'))
    
    model_stats.append(plot_model_group(MODELS_EN, 'whisper_en', 'Whisper English-Only', True))

    model_stats.append(plot_model_group(W2V_MODELS, 'wav2vec', 'Wav2Vec2', True))

    model_stats.append(plot_model_group(CANARY, 'canary', 'Canary', True))

    latex_table(model_stats)


if __name__ == '__main__':
    main()
