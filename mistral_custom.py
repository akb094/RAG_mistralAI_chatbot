# mistral_custom.py
from langchain.llms.base import LLM
from mistralai.client import MistralClient
from pydantic import Field, ConfigDict
from typing import Optional, List, Mapping, Any
import os

class MistralAI(LLM):
    model: str = Field(default="mistral-tiny")
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)
    model_config = ConfigDict(extra='allow')

    def __init__(self, **data):
        super().__init__(**data)
        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        self.client = MistralClient(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "mistral"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
