import glob
import logging
import os
import time
from datetime import datetime
from threading import Lock

from flask import Flask
from flask import request, jsonify

from generator import ExLlamaGenerator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer

print(f"Loading model")
# Directory containing config.json, tokenizer.model and safetensors file for the model
model_directory = "/data/model"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

config = ExLlamaConfig(model_config_path)  # create config from config.json
config.model_path = model_path  # supply path to model weights file

model = ExLlama(config)  # create ExLlama instance and load the weights
print(f"Model loaded: {model_path}")

tokenizer = ExLlamaTokenizer(tokenizer_path)  # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)  # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

app = Flask(__name__)
request_lock = Lock()

logging.basicConfig(level=logging.INFO)


@app.before_request
def log_request():
    app.logger.info(f"Route accessed: {request.path}, Method: {request.method}")


@app.route('/chat/completions', methods=['POST'])
def completions_simple():
    with request_lock:
        data = request.json

        print(f"Message received")
        messages = data['messages']

        input_model = data.get('model', 'exllama')
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 1024)
        presence_penalty = data.get('presence_penalty', 1.2)

        prompt = generate_prompt(messages)
        print(prompt)

        generator.settings.token_repetition_penalty_max = presence_penalty
        generator.settings.token_repetition_penalty_sustain = config.max_seq_len
        generator.settings.temperature = temperature
        generator.settings.top_p = top_p
        generator.settings.top_k = 40
        generator.settings.typical = 0.0

        output_content = generator.generate_simple(prompt, max_new_tokens=max_tokens, stop_words=["user:", "[INST]", "[/INST]", "[inst]", "[/inst]", "<s>", "</s>", "<S>", "</S>", "<<SYS>>", "<</SYS>>", "<<sys>>", "<</sys>>", "[/", ". >", ". <", "</"])
        print(output_content)

        messages.append({
            "role": "assistant",
            "content": output_content
        })

        response = {
            "id": "generated",
            "object": "chat.completion",
            "created": int(time.mktime(datetime.now().timetuple())),
            "model": input_model,
            "choices": [{
                "index": 0,
                "message": messages[-1],
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return jsonify(response)


def generate_prompt(messages):
    system_prompt = "You are a chat assistant, having a conversation with a user about anything. You are given the " \
                    "history of the conversation. Continue the conversation by replying back to the user as the " \
                    "assistant."

    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    for i, message in enumerate(messages):
        user_msg_role = message['role']
        user_msg_content = message['content']

        if user_msg_role == "user":
            if i != len(messages) - 1 or len(messages) == 1:
                prompt += f"{user_msg_role}: {user_msg_content} [/INST] "
            else:
                prompt += f"[INST] {user_msg_role}: {user_msg_content} [/INST]"
        else:
            if i != len(messages) - 1:
                prompt += f"{user_msg_role}: {user_msg_content} </s> <s> "
            else:
                prompt += f"{user_msg_role}: {user_msg_content} "

    prompt += " assistant: "

    return prompt


host = "0.0.0.0"
port = 8080
print(f"Server started on address {host}:{port}")

if __name__ == '__main__':
    from waitress import serve

    serve(app, host=host, port=port)
