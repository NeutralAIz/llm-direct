import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
import os
from superagi.llms.openai import OpenAi

class OpenAIDirectSchema(BaseModel):
    system: str = Field(
        "You are a helpful assistant",
        description="How the AI should act - examples: You are a data scientist and you..., You are a software architect creating a diagram, You are a writer building a story...",
    )
    message: str = Field(
        ...,
        description="The message you would like the model to respond to",
    )
    data: Optional[str] = Field(
        None,
        description="Structured data you would like to add in a seperate message, in the same thread",
    )
    model: Optional[str] = Field(
        "gpt-3.5-turbo",
        description="Which OpenAI Model to use:\nname: gpt-3.5-turbo description: 4k length, cheapest, default\nname: gpt-4 description: 8k length, smartest\nname: gpt-3.5-turbo-16k description: 16k length, longest",
    )
    
class OpenAIDirectTool(BaseTool):
    name = "OpenAI Direct Call Tool"
    description = (
        "Make a call directly to the OpenAI GPT series of models."
    )
    args_schema: Type[OpenAIDirectSchema] = OpenAIDirectSchema

    def _execute(self, system: str, message: str, data: str, model: str):
        # Retrieve the API key from the environment variable or, if not set, the application's config
        api_key = os.environ.get("OPENAI_API_KEY", None) or get_config("OPENAI_API_KEY", "")

        if not api_key:
            raise Exception("OpenAI API Key not found in environment variables or application configuration.")

        # Initialize the OpenAi class with the API key and chosen model
        openai_api = OpenAi(api_key=api_key, model=model)

        # Package the system and message inputs into the messages list format as needed by the Chat API
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]

        # If there is additional structured data, add it to the messages list as a user message
        if data:
            messages.append({"role": "user", "content": data})

        # Perform the API call and return the results
        result = openai_api.chat_completion(messages)
        
        if "error" in result:
            # There was an error with the API call, raise an exception
            raise Exception(f"Error running OpenAI API: {result['error']}")
        
        return result['content']