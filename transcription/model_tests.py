from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
from jiwer import wer
from scipy import signal

from huggingface_hub import login

login()

# load model and tokenizer
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Bigger model:
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

# cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", streaming=True)
# print(next(iter(cv_16)))



N_SAMPLES = 1

# load dummy dataset and read soundfiles
# "en" gets english data
cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", streaming=True)
ds = [next(iter(cv_16)) for i in range(N_SAMPLES)]
true_words = [s['sentence'] for s in ds]
accents = [s['accent'] for s in ds]

# Convert to proper sampling rate
sr = processor.feature_extractor.sampling_rate
audio_sr = ds[0]["audio"]["sampling_rate"]
assert audio_sr > sr, 'Sampling rate of input audio must be larger than models sampling rate to be converted properly'
audio_arr = ds[0]["audio"]["array"]
if audio_sr != sr:
  sr_ratio = audio_sr / sr
  if sr_ratio.is_integer():
    print('Decimating...')
    audio_arr = signal.decimate(audio_arr, int(sr_ratio))
  else:
    samples_resamp = int(len(audio_arr) / audio_sr * sr)
    audio_arr = signal.resample(audio_arr, samples_resamp)

# tokenize
input_values = processor(audio_arr, sampling_rate=sr, return_tensors="pt", padding="longest").input_values  # Batch size 1



# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)



print('*'*50)
print(transcription[0])
print('-'*50)
print(true_words[0])
print('-'*50)
print(accents[0])
print('*'*50)
