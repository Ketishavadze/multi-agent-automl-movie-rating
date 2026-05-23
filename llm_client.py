import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_llm(system_prompt, user_prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=system_prompt,
        input=user_prompt,
    )

    return response.output_text