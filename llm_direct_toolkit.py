from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseTool, BaseToolkit
from openai_direct import OpenAIDirectTool


class LLMDirectToolkit(BaseToolkit, ABC):
    name: str = "LLM Direct Toolkit"
    description: str = "Communicate directly with an LLM model."

    def get_tools(self) -> List[BaseTool]:
        return [
            OpenAIDirectTool(),
        ]

    def get_env_keys(self) -> List[str]:
        return []
