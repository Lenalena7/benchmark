import torch
from pathlib import Path
import sys
sys.path.append('/data/gas/tsao_data/radioactive/src/radioa/utils/sam3')
from sam3.model_builder import build_sam3_image_model

model = build_sam3_image_model(
    checkpoint_path='/data/gas/tsao_data/radioactive/src/radioa/model/SAM3.py',
    bpe_path='/data/gas/tsao_data/radioactive/src/radioa/utils/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
)

# 印出每層的 dtype
for name, param in model.named_parameters():
    print(name, param.dtype)
