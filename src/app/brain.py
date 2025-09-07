import os
from src.app.settings import Settings

settings = Settings()
client = settings.client
model = settings.brain_model


def brain_reply(input_text: str): 
    completion = client.chat.completions.create(
        model= model,
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant who answers questions briefly."
            },

            {
                "role": "user",
                "content": input_text,
            },
        ]
    )

    brain_output = completion.choices[0].message
    print(f"Brain Reply: {brain_reply}\n")
    return brain_output.content


if __name__ == "__main__":
    test_input = "Tell me about the capital of France?"
    print(brain_reply(test_input))
    