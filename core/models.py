import openai
import vertexai
from llama_index.llms.llama_cpp import LlamaCPP
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
        self.model = LlamaCPP(
            # URL to download the model in GGUF format
            model_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_0.gguf",
            # controls the randomness of the text generation. A lower temperature like 0.1 makes the model output more deterministic and focused, while a higher temperature makes it produce more varied and creative text.
            temperature=0.1,
            # Maximum number of tokens that model will generate in response to prompt
            max_new_tokens=300,
            # the number of tokens the model can consider when generating new text. For the Lama2 the maximal number is 4096
            context_window=3900,
            # set to at least 1 layer to use GPU
            # greate number may speed up computations bu also lead to "out of memeory" errors
            model_kwargs={"n_gpu_layers": 32},
            verbose=True,
        )

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        super().debug(f">>> {__class__.__name__}: {prompt}")

        completion = self.model.complete(prompt)

        completion = completion.strip()
        super().debug(f"{__class__.__name__} >>>: {completion}")
        return completion
