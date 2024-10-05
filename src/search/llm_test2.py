
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/ssamot/Downloads/codegeex4-all-9b-IQ2_M.gguf"


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex4-all-9b",  gguf_file=MODEL_PATH,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/codegeex4-all-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "write a quick sort"}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True ).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=256)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
