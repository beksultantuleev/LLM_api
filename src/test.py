# to test a Pythia model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half()

input_text = '<|prompter|>How are you doin?<|endoftext|><|assistant|>'

input_ids = tokenizer.encode(input_text, return_tensors='pt')

# sample_output = model.generate(input_ids, max_length=256, do_sample=True,
#                                early_stopping=True,
#                                num_return_sequences=1)

sample_output = model.

output_text  = tokenizer.decode(sample_output[0], skip_special_tokens=False)

print(output_text)


######
import requests
import json
import colorama

SERVER_IP = "localhost"
URL = f"http://{SERVER_IP}:5000/generate"

USERTOKEN = "<|prompter|>"
ENDTOKEN = "<|endoftext|>"
ASSISTANTTOKEN = "<|assistant|>"

def prompt(inp):
    data = {"text": inp}
    headers = {'Content-type': 'application/json'}

    response = requests.post(URL, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return "Error:", response.status_code
    
history = ""
while True:
    inp = input(">>> ")
    context = history + USERTOKEN + inp + ENDTOKEN + ASSISTANTTOKEN
    output = prompt(context)
    history = output
    print(history)
    just_latest_asst_output = output.split(ASSISTANTTOKEN)[-1].split(ENDTOKEN)[0]
    # color just_latest_asst_output green in print:
    print(colorama.Fore.GREEN + just_latest_asst_output + colorama.Style.RESET_ALL)


   