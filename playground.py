import json
from transformers import GPTNeoXConfig
from modelling_gpt_neox_moe import GPTNeoXForCausalLM
import transformers
from safetensors.torch import load_model, load_file


with open("/mnt/scratch/bborisov/models/MoE_MDEL/config.json", "r") as fp:
  config = GPTNeoXConfig(**json.load(fp))
model = GPTNeoXForCausalLM(config).to(device = 'cpu')
# for key, param in model.named_parameters():
#   print(key)
state_dict = load_file("/mnt/scratch/bborisov/models/MoE_MDEL/model-00001-of-00001.safetensors")
# print("\n".join(state_dict.keys()))
load_model(model, "/mnt/scratch/bborisov/models/MoE_MDEL/model-00001-of-00001.safetensors", strict=False)
# print("model initialised")
tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
pipe = transformers.TextGenerationPipeline(
    model=model, tokenizer=tokenizer, device="cuda"
)
print("\n".join(state_dict.keys() - model.state_dict().keys()))
# print()
# print("\n".join(model.state_dict().keys()) - state_dict.keys())
print(pipe(
"What is the formula for the area of a circle?",            
max_new_tokens=128,
do_sample=True,
temperature=0.3,
top_k=50,
top_p=0.9,
return_full_text=False,
)[0]['generated_text'])