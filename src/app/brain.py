import os
from src.app.settings import Settings

settings = Settings()
client = settings.client
model = settings.brain_model


def chat_brain(input_text: str):
    
    completion = client.chat.completions.create(
        model= model,
        messages=[
            {
                "role": "user",
                "content": input_text,
            }
        ],
    )

    brain_output = completion.choices[0].message

    return brain_output.reasoning, brain_output.content


if __name__ == "__main__":
    test_input = "Hello, how are you?"
    print(chat_brain(test_input))
