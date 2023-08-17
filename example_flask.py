import glob
import os
import time

from flask import Flask
from flask import request, jsonify
from threading import Lock
from datetime import datetime

from generator import ExLlamaGenerator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer

print(f"Starting API")
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


@app.route('/v1/chat/completions', methods=['POST'])
def completions():
    with request_lock:
        data = request.json

        messages = data['messages']

        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 1024)
        presence_penalty = data.get('presence_penalty', 1.2)

        # System prompt
        system_prompt = """
        <s>[INST] <<SYS>>
        You will provide dialogue to the conversation.
        <</SYS>>
        """

        # Start with the system prompt for the first message
        if len(messages) == 1:
            prompt = system_prompt
            user_msg = messages[0]['role']
            model_answer = messages[0]['content']
            prompt += f"{user_msg} [/INST] {model_answer}</s>"
        else:
            # For subsequent messages, append the interaction
            prompt = ""
            for message in messages:
                user_msg = message['role']
                model_answer = message['content']
                prompt += f"<s>[INST] {user_msg} [/INST] {model_answer}</s>"

        print(prompt)

        generator.settings.token_repetition_penalty_max = presence_penalty
        generator.settings.token_repetition_penalty_sustain = config.max_seq_len
        generator.settings.temperature = temperature
        generator.settings.top_p = top_p
        generator.settings.top_k = 40
        generator.settings.typical = 0.25

        output_content = generator.generate_simple(prompt, max_new_tokens=max_tokens)

        messages.append({
            "role": "assistant",
            "content": output_content
        })

        response = {
            "id": "generated",
            "object": "chat.completion",
            "created": int(time.mktime(datetime.now().timetuple())),
            "model": "gpt-3.5-turbo-0613",
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


host = "0.0.0.0"
port = 8080
print(f"Server started on address {host}:{port}")

if __name__ == '__main__':
    from waitress import serve

    serve(app, host=host, port=port)
