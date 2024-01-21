import os
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed
from huggingface_hub import InferenceClient

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_KEY = os.environ["OPENAI_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]


GPT3 = "gpt-3.5-turbo-1106"
GPT4 = "gpt-4-1106-preview"
ZEPHYR = "meta-llama/Llama-2-7b-chat-hf"


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


@retry(wait=wait_fixed(15), stop=stop_after_attempt(4))
def llm(user_prompt, model, temperature=0.3):
    model_kwargs = {"temperature": temperature}
    user_prompt = user_prompt[:3700]

    client = OpenAI(api_key=OPENAI_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        stream=False,
        **model_kwargs,
    )

    output = response.choices[0].message.content

    return output


def hf_llm(text, model, params={"max_new_tokens": 500, "temperature": 0.3}):
    prompt = f"<s>[INST]{text}[/INST]"
    client = InferenceClient(model=model, token=HF_TOKEN)

    output = client.text_generation(prompt, **params)
    return output


decomposition_prompt_template = (
    lambda primary_goal, personalization, history: f"""I want to accomplish the main goal of {primary_goal}
To better assist me, please break down the problem into sub-problems.
Each sub-problem should help me to solve the original problem.
Make it so that each sub-problem is not trival and can be helpful.
Take my context and personalizeation cues to personalize the sub-problems.
Make sure each sub-problem is concise and less than 15 words.

Personalization Cue: {personalization}
My Context: {history}

Return a numbered list with a maximum of 8 sub-problems, each on a new line.

Output: """
)

options_prompt_template = (
    lambda primary_goal, task, personalization, history: f"""
Here is one of the sub-query to help answer the main query.
Go into details to help me with the sub -query.
Show me some options to personalize and choose from.
Be concrete and make sure the options are valid choices to finish the task in sub-query.
When coming up with options, make sure they are diverse and representative of multiple demographics, cultures, and viewpoints.
Output a numbered list of options for me to choose from, with each option on a new line. Provide helpful details.

Primary-query: {primary_goal}
Sub-query: {task}
Personalization Cue: {personalization}
My Context: {history}

Output:"""
)

summarize_prompt_template = (
    lambda primary_goal, history: f"""I want to solve the problem in the primary-query.
I've loaded all of my research in the history so you can see what I've done so far. 
The history is in the schema {{problem2solve: solutions}}.
Please synthesize my history into a final report about the primary-query.
Make sure to remove any duplicate information if it exists in the history.

Primary-query: {primary_goal}
History: 
{history}

Output:"""
)


def decomposition(primary_goal, personalization, history):
    prompt = decomposition_prompt_template(primary_goal, personalization, history)
    output = llm(prompt, GPT3)
    output = [x.lstrip("0123456789. ") for x in output.split("\n")]
    return output


def options(primary_goal, task, personalization, history):
    prompt = options_prompt_template(primary_goal, task, personalization, history)
    output = llm(prompt, GPT3)
    output = [x.lstrip("0123456789. ") for x in output.split("\n")]
    return output


def summarize(primary_goal, history):
    prompt = summarize_prompt_template(primary_goal, history)
    output = llm(prompt, GPT3)
    return output
