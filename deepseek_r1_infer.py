import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DeepseekV3ForCausalLM



self_config = {
    "num_hidden_layers":4
}

model = DeepseekV3ForCausalLM.from_pretrained("/gcsp_afs/gongqiwei/DeepSeek-R1", torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=False, **self_config)
tokenizer = AutoTokenizer.from_pretrained("/gcsp_afs/gongqiwei/DeepSeek-R1")
input = tokenizer("你好，DeepSeek-V3！", return_tensors="pt")
device = model.device
input_ids= input["input_ids"].to(device)
attention_mask = input["attention_mask"].to(device)
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
print(tokenizer.decode(outputs[0]))
