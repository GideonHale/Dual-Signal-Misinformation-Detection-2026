import agents.ollama_agent as ag
from roles import role_titles, roles
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model1 = "mixtral:8x7b"
model2 = "gpt-oss:120b"
model3 = "qwen2.5:72b"

role = "journalist"

joseph = ag.Agent(
    name="Joseph",
    system_prompt=roles[role][0],
    client=client,
    model=model1
)

steven = ag.Agent(
    name="Steven",
    system_prompt=roles[role][1],
    client=client,
    model=model2
)

benjamin = ag.Agent(
    name="Benjamin",
    system_prompt=roles[role][2],
    client=client,
    model=model3
)

christopher = ag.Agent(
    name="Christopher",
    system_prompt=roles[role][3],
    client=client,
    model=model1
)

elijah = ag.Agent(
    name="Elijah",
    system_prompt=roles[role][4],
    client=client,
    model=model2
)

summarizer = ag.Agent(
    name="Summarizer",
    system_prompt="You are a neutral observer. Summarize the debate by analyzing each delineated response and identifying the core arguments.",
    client=client,
    model=model3
)

judge = ag.Agent(
    name="Judge",
    system_prompt="You are an unbiased judge. Give a verdict that measures the reliability of the article based on the other agents' responses. The verdict should be an integer from 0 to 5 (where 0 is completely unreliable and 5 is completely reliable) and a brief explanation of your reasoning.",
    client=client,
    model=model3
)

extractor = ag.Agent(
    name="Extractor",
    system_prompt="You are an unbiased extractor. Find the numerical score near the beginning of the prompt and return only that. Return it as an integer from 0 to 5. For example, if you get '5 / 5' return only 5. If you get '3' return only 3. If you get '4 / 5 and other text' return 4. If you get 'The score is 0 because ..' or something similar, return only 0. If you get 'Verdict: 4 / 5' return 4. If you get any other number outside of that range return it so the error can be caught somewhere else.",
    client=client,
    model=model1
)