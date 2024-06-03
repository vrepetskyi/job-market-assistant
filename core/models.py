import openai
import vertexai
from vertexai.generative_models import GenerativeModel


class Model:
    def debug(self, message: str) -> None:
        if self.debug:
            print(message)

    def complete(self, prompt: str) -> str: ...


class ChatGPT(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.client = openai.Client(api_key="")

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().debug(f">>> {__class__.__name__}: {prompt}")

        response = self.client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=512,
        )

        completion = (
            response.choices[0].text
            if isinstance(response, openai.types.Completion)
            else ""
        )

        completion = completion.strip()
        super().debug(f"{__class__.__name__} >>>: {completion}")
        return completion


class Gemini(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        vertexai.init(project="linkedln-advisor", location="europe-central2")
        self.model = GenerativeModel(
            model_name="gemini-1.0-pro-002",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 200,
            },
        )

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().debug(f">>> {__class__.__name__}: {prompt}")

        completion = self.model.generate_content(prompt).text

        completion = completion.strip()
        super().debug(f"{__class__.__name__} >>>: {completion}")
        return completion


class Llama(Model):
    def __init__(self, debug=False) -> None:
        self.debug = debug

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().debug(f">>> {__class__.__name__}: {prompt}")

        completion = ""

        completion = completion.strip()
        super().debug(f"{__class__.__name__} >>>: {completion}")
        return completion
