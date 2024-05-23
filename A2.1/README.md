- [Assignment PDF](./A2-1.pdf)
- [Prompt Generator Class Implementation](./prompt_gen.py)

## Running Gemma
```python
# Setup the environment
!pip install -q -U immutabledict sentencepiece 
!git clone https://github.com/google/gemma_pytorch.git
!mkdir /kaggle/working/gemma/
!mv /kaggle/working/gemma_pytorch/gemma/* /kaggle/working/gemma/

import sys 
sys.path.append("/kaggle/working/gemma_pytorch/") 
from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch

# Load the model
VARIANT = "7b-it-quant" 
MACHINE_TYPE = "cuda" 
weights_dir = '/kaggle/input/gemma/pytorch/7b-it-quant/2' 

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT

device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
  model.load_weights(ckpt_path)
  model = model.to(device).eval()


# Use the model

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

sample = {
    "question":"What is the sum of week(s) with an attendance of 30,751?",
    "table":{
        "cols":["Week","Date","Opponent","Result","Attendance"],
        "rows":[
            ["1","August 6, 1973","San Francisco 49ers","L 27\u201316","65,707"],
            ["2","August 11, 1973","at Los Angeles Rams","T 21\u201321","54,385"],
            ["3","August 19, 1973","vs. Cincinnati Bengals at Columbus, Ohio","W 24\u20136","73,421"],
            ["4","August 25, 1973","vs. Atlanta Falcons at Knoxville","W 20\u201317","40,831"],
            ["5","September 1, 1973","Detroit Lions","L 16\u201313","64,088"],
            ["6","September 8, 1973","vs. New York Giants at Akron","L 21\u201310","30,751"]
        ],
        "types":["real","text","text","text","real"],
        "caption":"Exhibition schedule"
    }
}

prompt_gen = PromptGenerator()
prompt = prompt_gen.create_prompt(sample)

gen_text = model.generate(
    USER_CHAT_TEMPLATE.format(prompt=prompt),
    device=device,
    output_len=prompt_gen.output_len,
    temperature=prompt_gen.temperature,
    top_p=prompt_gen.top_p,
)
answer = prompt_gen.post_process(gen_text)
```