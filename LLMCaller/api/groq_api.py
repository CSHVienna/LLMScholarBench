import os
from groq import Groq

class GroqAPI:
    def __init__(self, config):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.config = config

    def generate_response(self, prompt):
        # Simply generate a response using the prompt provided
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.config['system_message']},
                {"role": "user", "content": prompt}
            ],
            model=self.config['model'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            top_p=self.config['top_p'],
            stop=self.config['stop'],
            stream=self.config['stream']
        )

        return chat_completion
