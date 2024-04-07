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


def plot_stat(x, y_list, ci_u_list, ci_l_list, accents, out_path, stat, model_sizes, model):
    fig, ax = plt.subplots()
    for y, cu, cl, a in zip(y_list, ci_u_list, ci_l_list, accents):
        ax.plot(x,y, label=a)
        # ax.fill_between(x, cl, cu, color='b', alpha=.1)
    plt.title(STATS_MAP[stat] + f" for {model}")
    plt.xlabel('Model Size (Number of Parameters)')
    plt.ylabel(f'Mean {STATS_MAP[stat]}')
    plt.xticks(list(model_sizes.values()), labels=[f'{s}M' for m, s in model_sizes.items()], rotation=45)

    # pos = ax.get_xaxis().majorTicks[-1].label1.get_position()
    # ax.get_xaxis().majorTicks[-1].label1.set_position((pos[0] - 5000, pos[1]))
    
    ax.get_xaxis().majorTicks[-1].label1.set_horizontalalignment('right')
    #ax.get_xaxis().majorTicks[-2].set_pad(3) # .label1.set_horizontalalignment('right')
    # ax.get_xaxis().majorTicks[-2].label1.set_horizontalalignment('left')

    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path)


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

        plot_stat(x, y_list, ci_u_list, ci_l_list, accent_groups, out_path, s, model_group, model_name)

    return model_stats

 
def latex_table(model_stats):

    stat_choice = 'jaro_winkler'
    accent_groups = list(model_stats[list(model_stats.keys())[0]].keys())
    models = list(model_stats.keys())
    print(' & '.join(['\\textbf{Model}'] + ['\\textbf{' + m + '}' for m in models]) + ' \\\\')
    for ag in accent_groups:
        row = [ag]
        for m in models:
            row.append(str(model_stats[m][ag][stat_choice][0].__round__(5)))

        print(' & '.join(row).replace('_', '\\_') + ' \\\\')


def main():
    model_stats = plot_model_group(MODELS, 'whisper', 'Whisper')
    # print('Whisper')
    # latex_table(model_stats)
    
    # model_stats = plot_model_group(MODELS_EN, 'whisper_en', 'Whisper English-Only', True)
    # # print('Whisper English')
    # # latex_table(model_stats)

    # model_stats = plot_model_group(W2V_MODELS, 'wav2vec', 'Wav2Vec2', True)
    # latex_table(model_stats)

    model_stats = plot_model_group(CANARY, 'canary', 'Canary', True)
    latex_table(model_stats)


if __name__ == '__main__':
    main()
