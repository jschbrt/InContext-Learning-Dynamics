"""
This file contains the helper functions to generate text from different LLMs.
"""
import os
import time
import anthropic
import openai
import numpy as np
from dotenv import load_dotenv
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#! 
def get_llm(engine, temp=0.0, max_tokens=1, arms=('F', 'J'), with_suffix=False):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    Q_, A_ = '\n\nQ:', '\n\nA:'
    if engine.startswith("gpt"):
        # load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY{2 if engine == 'gpt-4' else ''}")
        load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY")
        llm = GPT4LLM((gpt_key, engine))
    elif engine.startswith("claude"):
        load_dotenv(); anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        llm = AnthropicLLM((anthropic_key, engine))
        Q_, A_ = anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT
    elif engine.startswith("n-claude"):
        load_dotenv(); anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        llm = NewAnthropicLLM((anthropic_key, engine))
        Q_, A_ = anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT
    elif engine.startswith("llama-2"):
        load_dotenv(); hf_key = os.getenv(f"HF_API_KEY")
        llm = HF_API_LLM((hf_key, engine, max_tokens, temp))
    else:
        print('No key found')
        llm = DebuggingLLM(arms)
    return llm, Q_, A_

class LLM:
    def __init__(self, llm_info):
        self.llm_info = llm_info

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        raise NotImplementedError

class DebuggingLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("DEBUGGING MODE")
        info = llm_info
        if info[0] == 'not a 2 armed bandit':
            self.random_fct = info[1]
        else:
            arm1 = info[0]
            arm2 = info[1]
            self.random_fct = lambda : arm1 if np.random.rand() < 0.5 else arm2

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        if arms:
            if  arms[0] == 'not a 2 armed bandit': 
                random_fct = arms[1]
            else:
                random_fct = lambda : arms[0] if np.random.rand() < 0.5 else arms[1]
        else:
            random_fct = self.random_fct
        return random_fct()

class GPT4LLM(LLM):
    def __init__(self, llm_info):
        self.gpt_key, self.engine = llm_info
        openai.api_key = self.gpt_key

    def generate(self, text, temp=0, max_tokens=1, arms=None):
        text = [{"role": "user", "content": text}]  
        time.sleep(1) # to avoid rate limit error which happens a lot for gpt4
        for iter in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model = self.engine,
                    messages = text,
                    max_tokens = max_tokens,
                    temperature = temp
                )
                return response.choices[0].message.content.replace(' ', '')
            except:
                time.sleep(3**iter)
                if iter == 5:
                    import ipdb; ipdb.set_trace()

class NewAnthropicLLM(LLM):
    def __init__(self, llm_info):
        self.anthropic_key, self.engine = llm_info
        self.engine = self.engine.replace('n-', '')

    def generate(self, instruction, question, temp=0.0, max_tokens=1, arms=None, wrong_text=None, wrong_action=None):

        retries = 0
        max_retries = 20
        try:
            c = anthropic.Anthropic(api_key=self.anthropic_key)

            if wrong_text == None:
                response = c.messages.create(
                    system = instruction,
                    messages=[{"role": "user", "content": question},
                              {"role": "assistant", "content": "Machine"}],
                    model=self.engine,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            else: 
                response = c.messages.create(
                    system = instruction,
                    messages=[{"role": "user", "content": question},
                            {"role": "assistant", "content": "Machine "+wrong_action},
                            {"role": "user", "content": wrong_text},
                            {"role": "assistant", "content": "Machine"}],
                    model=self.engine,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            c.close()
            r = response.content[0].text.strip(" .,\n:")
            return r
        except Exception as e:
            print(e)
            retries += 1
            if retries < max_retries:
                time.sleep(3)
                return self.generate(instruction, question, temp, max_tokens, arms)
            else:
                print("Too many retries")
                raise e

class AnthropicLLM(LLM):
    def __init__(self, llm_info):
        self.anthropic_key, self.engine = llm_info

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):

        retries = 0
        max_retries = 20
        try:
            c = anthropic.Anthropic(api_key=self.anthropic_key)

            response = c.completions.create(
                prompt = anthropic.HUMAN_PROMPT + text,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=self.engine,
                temperature=temp,
                max_tokens_to_sample=max_tokens,
            ).completion.replace(' ', '')
            c.close()
            return response
        except Exception as e:
            print(e)
            retries += 1
            if retries < max_retries:
                time.sleep(3)
                return self.generate(text, temp, max_tokens, arms)
            else:
                print("Too many retries")
                raise e

class HF_API_LLM(LLM):
    def __init__(self, llm_info):
        hf_key, engine, max_tokens, temperature = llm_info
        padtokenId = 50256 # Falcon needs that to avoid some annoying warning

        # Authenticate
        from huggingface_hub import notebook_login
        notebook_login(hf_key)

        if engine.startswith('llama-2'):
            #Change llama-2-* to meta-llama/Llama-2-*b-hf
            if 'chat' in engine:
                engine = 'meta-llama/L' +engine[1:].replace('-chat', '') + 'b-chat-hf'
            else:
                engine = 'meta-llama/L' +engine[1:] + 'b-hf'
        else:
            print("Wrong engine name for HF API LLM")
            raise NotImplementedError

        print(engine)
        self.tokenizer = AutoTokenizer.from_pretrained(engine, token=hf_key)
        self.model = AutoModelForCausalLM.from_pretrained(engine, device_map="auto", torch_dtype=torch.bfloat16, token=hf_key)
        self.max_tokens = max_tokens
        # Adapt pipeline to set temperature to 0
        self.temperature = temperature +1e-6
    
    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        model_outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        response = self.tokenizer.decode(model_outputs[0][model_inputs.input_ids[0].shape[0]:])
        return response
