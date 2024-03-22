```
module load conda
conda create --name asr_bias python=3.10
eval "$(conda shell.bash hook)"

conda activate asr_bias

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```