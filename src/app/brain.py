import os
from src.app.settings import Settings

settings = Settings()
client = settings.client
model = settings.brain_model

# prompt= """
# You are a helpful conversational assistant that provides clear, plain text answers suitable for speech. 
# Do not use asterisks (*), dashes (—), bullet points, emojis, or any other formatting symbols. 
# Do not use abbreviations or short forms such as "e.g." or "i.e."; always write them out in full. 
# Keep your responses natural, complete, and easy to read aloud.
# """

prompt = """
You are a professional technical interviewer conducting a virtual interview for a Data Science role. Your goal is to evaluate the candidate's skills, knowledge, and problem-solving ability in areas relevant to data science. 

Instructions:
1. Begin by greeting the candidate and explaining that this is a technical interview for a Data Science role.
2. Ask open-ended questions covering key areas such as:
   - Python programming and data manipulation (pandas, numpy)
   - SQL and database querying
   - Statistics and probability
   - Machine learning concepts and algorithms
   - Data cleaning, preprocessing, and feature engineering
   - Data visualization and storytelling
   - AI/ML model evaluation metrics
   - Cloud or Big Data tools (optional, depending on candidate level)
3. After each candidate answer, ask a follow-up question that probes deeper or asks for a practical example.
4. Be professional, polite, and encouraging, but critical where appropriate.

"""
history = [
            {
                "role": "system", 
                "content": prompt
            }
    ]

def brain_reply(input_text: str): 

    history.append({"role": "user", "content": input_text})

    completion = client.chat.completions.create(
        model= model,
        messages=history,
    )

    assistant_reply = completion.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply


if __name__ == "__main__":
    
    print(brain_reply("What’s the capital of France?"))
    print(brain_reply("And what’s its population?"))
    print(brain_reply("Which river flows through it?"))
    print(brain_reply("what city are we talking about?"))

    print("=*=" * 20)
    print(history)

    