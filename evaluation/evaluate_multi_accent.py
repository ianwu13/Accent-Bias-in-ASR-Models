import json
import requests
import time
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

from evaluator import Evaluator


API_TOKEN = 'FILL_LOCALLY'

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def get_subsentence_similarities(reference, candidates: list):
    payload = {
        "inputs": {
            "source_sentence": reference,
            "sentences": candidates
            }
        }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    # Handle potential API failure/rate limiting
    if response.status_code != 200:
        time.sleep(60)
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code != 200:
            return None

    return response.json()


# OLD VERSION
# def get_transcription_subsentence(sentence, transcription_tokens, front: bool):
#     if front:
#         candidates = [' '.join(transcription_tokens[:i]) for i in range(1, len(transcription_tokens)+1)]
#     else:
#         candidates = [' '.join(transcription_tokens[i:]) for i in range(len(transcription_tokens))]

#     sims = get_subsentence_similarities(sentence, candidates)

#     # handle bad request
#     if sims is None:
#         print(f'Failed Call: SENTENCE: "{sentence}", CANDIDATES: {candidates}')
#         return ''

#     return candidates[np.argmax(sims)]


# OLD VERSION
# def transcription_subsentence_index(sentence, transcription_tokens, front: bool):
#     sent_len = len(sentence.split())
#     if front:
#         candidates = [' '.join(transcription_tokens[:i]) for i in range(sent_len-SPREAD, sent_len+SPREAD)]
#     else:
#         candidates = [' '.join(transcription_tokens[-i:]) for i in range(sent_len-SPREAD, sent_len+SPREAD)]
#         print(candidates)

#     sims = get_subsentence_similarities(sentence, candidates)

#     # handle bad request
#     if sims is None:
#         print(f'Failed Call: SENTENCE: "{sentence}", CANDIDATES: {candidates}')
#         return ''

#     return np.argmax(sims)


def get_transcription_subsentence(sentence_1, sentence_2, transcription_tokens):

    candidates_1 = [' '.join(transcription_tokens[:i]) for i in range(len(transcription_tokens))]
    candidates_2 = [' '.join(transcription_tokens[i:]) for i in range(len(transcription_tokens))]

    sims_1 = get_subsentence_similarities(sentence_1, candidates_1)
    sims_2 = get_subsentence_similarities(sentence_2, candidates_2)

    max_sim_splt = np.argmax([s1 + s2 for s1, s2 in zip(sims_1, sims_2)])
    return candidates_1[max_sim_splt], candidates_2[max_sim_splt]


def identify_transcription_subsentences(df):
    transcription_subsentence_a = []
    transcription_subsentence_b = []

    for item in tqdm(df.iterrows()):
        row = item[1]

        sentence_1 = row['preprocessed_sentence_b'] if row['reversed'] else row['preprocessed_sentence_a']
        sentence_2 = row['preprocessed_sentence_a'] if row['reversed'] else row['preprocessed_sentence_b']
        transcription_tokens = row['preprocessed_transcriptions'].split()

        trsn_sub_1, trsn_sub_2 = get_transcription_subsentence(sentence_1, sentence_2, transcription_tokens)

        if row['reversed']:
            transcription_subsentence_b.append(trsn_sub_1)
            transcription_subsentence_a.append(trsn_sub_2)
        else:
            transcription_subsentence_a.append(trsn_sub_1)
            transcription_subsentence_b.append(trsn_sub_2)

    df['transcription_subsentence_a'] = transcription_subsentence_a
    df['transcription_subsentence_b'] = transcription_subsentence_b
    
    return df


def evaluate_transcription_set(df, evaluator):
    pred_a = df['transcription_subsentence_a'].tolist()
    ref_a = df['preprocessed_sentence_a'].tolist()

    wer_a = evaluator.wer(pred_a, ref_a)
    cer_a = evaluator.cer(pred_a, ref_a)
    jaro_winkler_a = evaluator.jaro_winkler(pred_a, ref_a)

    df['wer_a'] = wer_a
    df['cer_a'] = cer_a
    df['jaro_winkler_a'] = jaro_winkler_a

    pred_b = df['transcription_subsentence_b'].tolist()
    ref_b = df['preprocessed_sentence_b'].tolist()

    wer_b = evaluator.wer(pred_b, ref_b)
    cer_b = evaluator.cer(pred_b, ref_b)
    jaro_winkler_b = evaluator.jaro_winkler(pred_b, ref_b)

    df['wer_b'] = wer_b
    df['cer_b'] = cer_b
    df['jaro_winkler_b'] = jaro_winkler_b

    return df


def main():
    parser = argparse.ArgumentParser(description='Script to evaluate performance of a model across different accents')
    
    parser.add_argument('--transcriptions_path', type=str, help='Path to tsv file containing transcriptions')
    parser.add_argument('--output_path', type=str, help='Directory to write results to')
    
    args = parser.parse_args()

    df = pd.read_csv(args.transcriptions_path, sep='\t')

    df = identify_transcription_subsentences(df)
    # Save subsentences to temp location just incase
    df.to_csv(args.output_path.replace('.', '_tmp.'), sep='\t', index=False)

    evaluator = Evaluator()
    df = evaluate_transcription_set(df, evaluator)

    df.to_csv(args.output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
