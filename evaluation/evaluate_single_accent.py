import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from evaluator import Evaluator


def evaluate_transcription_set(df, evaluator):
    pred = df['preprocessed_transcriptions'].tolist()
    ref = df['preprocessed_sentence'].tolist()

    wer = evaluator.wer(pred, ref)
    cer = evaluator.cer(pred, ref)
    bertscore = evaluator.bertscore(pred, ref)
    jaro_winkler = evaluator.jaro_winkler(pred, ref)

    return {
        'wer': wer,
        'cer': cer,
        'bertscore_precision': bertscore['precision'],
        'bertscore_recall': bertscore['recall'],
        'bertscore_f1': bertscore['f1'],
        'jaro_winkler': jaro_winkler
    }


def main():
    parser = argparse.ArgumentParser(description='Script to evaluate performance of a model across different accents')
    
    parser.add_argument('--transcriptions_path', type=str, default='../data/cv16/sa_transcriptions.tsv',
        help='Path to tsv file containing transcriptions')
    parser.add_argument('--outputs_dir', type=str, default='results/sa_results',
        help='Directory to write results to')

    parser.add_argument('--bertscore_model', type=str, default='distilbert-base-uncased',
        help='Model to use in calculating BERTScore')
    
    args = parser.parse_args()

    transcriptions = pd.read_csv(args.transcriptions_path, sep='\t')
    evaluator = Evaluator(bert_model=args.bertscore_model, preload_bertscore_model=True)

    transcriptions_by_ag = {ag.replace('/', '_').replace('-', '_').replace('.', '_').replace(' ', '_'): rows for ag, rows in transcriptions.groupby('accent_group')}
    results = {ag: evaluate_transcription_set(rows, evaluator) for ag, rows in transcriptions_by_ag.items()}

    # Make sure output_dirs make sense
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    for ag in results.keys():
        tmp_pth = '/'.join([args.outputs_dir, ag])
        if not os.path.exists(tmp_pth):
            os.mkdir(tmp_pth)

    # record results in dataframe
    for ag, res in results.items():
        for metric, scores in res.items():
            transcriptions_by_ag[ag][metric] = scores
    results_df = pd.concat(list(transcriptions_by_ag.values()))
    # write results to tsv file
    results_path = '/'.join([args.outputs_dir, 'sa_transcriptions.tsv'])
    results_df.to_csv(results_path, sep='\t', index=False)

    # plot distributions of scores in histograms
    for ag, res in results.items():
        # Create subplots
        fig, axs = plt.subplots(6, figsize=(8, 10))

        # Plot histograms
        axs[0].hist(res['wer'], bins=30, color='skyblue')
        axs[0].set_title(f'Histogram for Distribution of WER for AccentGroup={ag}')

        axs[1].hist(res['cer'], bins=30, color='salmon')
        axs[1].set_title(f'Histogram for Distribution of CER for AccentGroup={ag}')

        axs[2].hist(res['bertscore_precision'], bins=30, color='lightgreen')
        axs[2].set_title(f'Histogram for Distribution of BERTScore_Precision for AccentGroup={ag}')

        axs[3].hist(res['bertscore_recall'], bins=30, color='gold')
        axs[3].set_title(f'Histogram for Distribution of BERTScore_Recall for AccentGroup={ag}')

        axs[4].hist(res['bertscore_f1'], bins=30, color='orchid')
        axs[4].set_title(f'Histogram for Distribution of BERTScore_F1 for AccentGroup={ag}')

        axs[5].hist(res['jaro_winkler'], bins=30, color='orchid')
        axs[5].set_title(f'Histogram for Distribution of Jaro-Winkler Distance for AccentGroup={ag}')

        # Adjust layout
        plt.tight_layout()
        
        plt.savefig('/'.join([args.outputs_dir, ag, 'histograms.png']))

    # TODO: Generate more statistics?


if __name__ == '__main__':
    main()
