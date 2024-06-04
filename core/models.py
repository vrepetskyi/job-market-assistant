import os

import openai
from vertexai.generative_models import GenerativeModel


class Model:
    def __init__(self, debug=False) -> None:
        self.debug = debug

    def info(self, message: str) -> None:
        if self.debug:
            print(message)

    def complete(self, prompt: str) -> str:
        raise NotImplementedError()


class ChatGPT(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().info(f">>> {self.__class__.__name__}: {prompt}")

        response = self.client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=512,
        )

        completion = response.choices[0].text

        completion = completion.strip()
        super().info(f"{self.__class__.__name__} >>>: {completion}")
        return completion


class Gemini(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.model = GenerativeModel(
            model_name="gemini-1.0-pro-002",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 200,
            },
        )

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().info(f">>> {self.__class__.__name__}: {prompt}")

        completion = self.model.generate_content(prompt).text or ""

        completion = completion.strip()
        super().info(f"{self.__class__.__name__} >>>: {completion}")
        return completion


class Llama(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.client = openai.Client(
            api_key=os.getenv("LLAMA_API_KEY"), base_url="https://api.llama-api.com"
        )

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().info(f">>> {self.__class__.__name__}: {prompt}")

        response = self.client.chat.completions.create(
            model="llama3-8b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        completion = response.choices[0].message.content or ""

        completion = completion.strip()
        super().info(f"{self.__class__.__name__} >>>: {completion}")
        return completion
