import glob
import os
import time
import torch
import logging

from flask import Flask
from flask import request, jsonify
from threading import Lock
from datetime import datetime

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

min_response_tokens = 4
extra_prune = 256
logging.basicConfig(level=logging.INFO)


def chat_with_exllama(messages, bot_name, username, max_response_tokens, break_on_newline=True):
    last_assistant_response = ""

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            res_line = bot_name + ":"
            res_tokens = tokenizer.encode(res_line)
            num_res_tokens = res_tokens.shape[-1]

            in_line = username + ": " + content.strip() + "\n"
            in_tokens = tokenizer.encode(in_line)
            in_tokens = torch.cat((in_tokens, res_tokens), dim=1)

            expect_tokens = in_tokens.shape[-1] + max_response_tokens
            max_tokens = config.max_seq_len - expect_tokens

            num_tokens = generator.gen_num_tokens()
            if num_tokens is not None and num_tokens >= max_tokens:
                generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)

            generator.gen_feed_tokens(in_tokens)
            generator.begin_beam_search()

            for i in range(max_response_tokens):
                if i < min_response_tokens:
                    generator.disallow_tokens([tokenizer.newline_token_id, tokenizer.eos_token_id])
                else:
                    generator.disallow_tokens(None)

                gen_token = generator.beam_search()

                if gen_token.item() == tokenizer.eos_token_id:
                    generator.replace_last_token(tokenizer.newline_token_id)

                num_res_tokens += 1
                text = tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
                new_text = text[len(res_line):]
                res_line += new_text

                if break_on_newline and gen_token.item() == tokenizer.newline_token_id: break
                if gen_token.item() == tokenizer.eos_token_id: break
                if res_line.endswith(f"{username}:"):
                    encoded_username = tokenizer.encode(f"{username}:")
                    plen = 0 if encoded_username is None else encoded_username.shape[-1]
                    generator.gen_rewind(plen)
                    break

            generator.end_beam_search()
            last_assistant_response = res_line

        elif role == "assistant":
            last_assistant_response = bot_name + ": " + content + "\n"

    cleaned_response = last_assistant_response.replace("assistant:", "").replace("user:", "").strip("\n ").strip("[/INST]")
    return cleaned_response


@app.before_request
def log_request():
    app.logger.info(f"Route accessed: {request.path}, Method: {request.method}")


@app.route('/chat/completions/bad', methods=['POST'])
def completions():
    with request_lock:
        data = request.json

        print(f"Message received")
        messages = data['messages']

        input_model = data.get('model', 'exllama')
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 1024)
        presence_penalty = data.get('presence_penalty', 1.2)

        generator.settings.token_repetition_penalty_max = presence_penalty
        generator.settings.token_repetition_penalty_sustain = config.max_seq_len
        generator.settings.temperature = temperature
        generator.settings.top_p = top_p
        generator.settings.top_k = 40
        generator.settings.typical = 0.0

        res = chat_with_exllama(messages, "assistant", "user", max_tokens, break_on_newline=False)
        response = {
            "id": "generated",
            "object": "chat.completion",
            "created": int(time.mktime(datetime.now().timetuple())),
            "model": input_model,
            "choices": [{
                "index": 0,
                "message": res,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return jsonify(response)


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

        output_content = generator.generate_simple(prompt, max_new_tokens=max_tokens, stop_tokens=["user:", "[/INST]", "</s>", "[/", ". >", ". <"])
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
    system_prompt = "You are an assistant, having a conversation with a user. You are given the history of the " \
                    "conversation. Reply back to the user only as the assistant, provide only a single message."

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
                prompt += f"{user_msg_role}: {user_msg_content} </s><s> "
            else:
                prompt += f"{user_msg_role}: {user_msg_content} "

    prompt += " assistant:"

    return prompt


host = "0.0.0.0"
port = 8080
print(f"Server started on address {host}:{port}")

if __name__ == '__main__':
    from waitress import serve

    serve(app, host=host, port=port)
