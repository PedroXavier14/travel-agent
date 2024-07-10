import os
from openai import OpenAI
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Vou viajar para Londres em 2024. Quero que faças um roteiro de viagem para mim."}
  ]
)

print(response.choices[0].message.content)