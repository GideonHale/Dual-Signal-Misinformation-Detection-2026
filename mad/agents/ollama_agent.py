from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any
from openai import OpenAI

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str
    author: str = 'Undefined'

@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str = "gpt-oss:120b"
    temperature: float = 0.7
    client: Any = field(default_factory=OpenAI)

    def respond(self, context: List[Message]) -> Message:
        """
        context: full conversation so far (without this agent's system prompt)
        returns: this agent's next assistant message
        """
        # 1. Start with the agent's unique persona
        messages = [{"role": "system", "content": self.system_prompt}]

        # 2. Convert transcript to OpenAI format
        for m in context:
            # If the message was written by THIS agent, it's an "assistant" role
            # If by anyone else, it's a "user" role (an external input)
            role = "assistant" if m.author == self.name else "user"
            content = f"{m.author}: {m.content}" if m.author != "System" else m.content
            
            messages.append({"role": role, "content": content})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        
        return Message(
            role="assistant", 
            content=completion.choices[0].message.content.strip(),
            author=self.name
        )