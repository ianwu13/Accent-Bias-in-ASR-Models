# ! pip install jiwer
import numpy as np
import jiwer

# WER calculation
# introduced during lecture, used in several evaluation papers in our drive
def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER). This metric is widely discussed, notably in Papers 1, 2, and 4.
    Args:
        reference (list of str): The correct list of words.
        hypothesis (list of str): The list of words to compare against the reference.
    Returns:
        float: The WER score.
    """
    # Split sentences into words if they're not already in list form
    if isinstance(reference, str):
        reference = reference.split()
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()
    
    # Initialization
    R = len(reference)
    H = len(hypothesis)
    cost_matrix = np.zeros((R+1, H+1))

    # Building the cost matrix
    for i in range(1, R+1):
        cost_matrix[i][0] = i
    for j in range(1, H+1):
        cost_matrix[0][j] = j

    for i in range(1, R+1):
        for j in range(1, H+1):
            if reference[i-1] == hypothesis[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i][j-1] + 1
                deletion = cost_matrix[i-1][j] + 1
                cost_matrix[i][j] = min(substitution, insertion, deletion)

    wer_score = cost_matrix[R][H] / R
    return wer_score

# example usage
reference_text = "test"
hypothesis_text = "test"
wer_score = calculate_wer(reference_text, hypothesis_text)
print(f"WER Score: {wer_score}")

# SeMaScore
# introduced in "Exploring practical Metrics to support Automatic Speech Recognition Evaluations" as a new metric
# as a placeholder due to the complexity of implementation
def calculate_semascore(reference, hypothesis):
    """
    Calculate SeMaScore - Introduced in Paper 2. This is a placeholder for the metric's calculation.
    Args:
        reference (str): The correct sentence.
        hypothesis (str): The sentence to compare against the reference.
    Returns:
        float: Placeholder value for SeMaScore.
    """
    pass  # need our own implementation if necessary

# print("SeMaScore:", calculate_semascore("reference", "hypothesis"))

# ——————————————————————————————————————————————————————————————————————————————————

def jiwer_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between a reference and a hypothesis.
    Args:
        reference (str): The correct sentence.
        hypothesis (str): The sentence to compare against the reference.

    Returns:
        float: The WER score.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation()
    ])
    wer_score = jiwer.wer(reference, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)
    return wer_score

def jiwer_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between a reference and a hypothesis.
    Args:
        reference (str): The correct sentence.
        hypothesis (str): The sentence to compare against the reference.

    Returns:
        float: The CER score.
    """
    cer_score = jiwer.cer(reference, hypothesis)
    return cer_score

# example usage
reference_sentence = "hello world"
hypothesis_sentence = "helo world"

wer = jiwer_wer(reference_sentence, hypothesis_sentence)
cer = jiwer_cer(reference_sentence, hypothesis_sentence)

print(f"Word Error Rate (WER): {wer}")
print(f"Character Error Rate (CER): {cer}")