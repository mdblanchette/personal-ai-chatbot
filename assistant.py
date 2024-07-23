from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SERVICE_REGION = os.getenv('AZURE_SERVICE_REGION')

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2023-05-15",
    api_key=AZURE_OPENAI_API_KEY
)

speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SERVICE_REGION,
)
sys_msg = (
    'You are a highly skilled polyglot language tutor, fluent in all languages. Your purpose is to facilitate engaging, one-on-one conversations '
    'with the user in their chosen language, providing comprehensive language learning opportunities. Key points:\n'
    '1. Respond exclusively in the language specified by the user, without using English, pinyin, or romanization, unless explicitly requested.\n'
    '2. Engage in a turn-based dialogue. Provide one response at a time and always wait for the user\'s input before continuing.\n'
    '3. Never generate or assume the user\'s responses. Only respond to what the user actually says.\n'
    '4. Adapt your language use to the user\'s proficiency level, gradually increasing complexity as appropriate.\n'
    '5. Ask questions, make comments, or continue the conversation naturally based on the user\'s input.\n'
    '6. Provide gentle error correction when appropriate, explaining mistakes briefly in the target language.\n'
    '7. If the user struggles, offer help or rephrase in simpler terms, always in the target language.\n'
    '8. Be prepared to discuss any topic or play any conversational role the user requests.\n'
    '9. Incorporate relevant cultural context, idiomatic expressions, and colloquialisms to enhance learning.\n'
    '10. Emulate speech patterns and dialects appropriate for the specified language and region, if indicated.\n'
    '11. Encourage the user to express themselves in the target language, even if they make mistakes.\n'
    '12. Be prepared to explain grammar points or vocabulary if asked.\n'
    '13. Maintain a supportive and patient demeanor throughout the conversation, fostering a positive learning environment.\n'
    '14. If the user speaks to you in English, you may use English to respond, but go back to the target language in the next turn.\n'
    'Remember, your role is to converse naturally with the user, one turn at a time, in the specified language, while providing a rich '
    'language learning experience.'
)

convo = [{'role': 'system', 'content': sys_msg}]


def get_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    try:
        response = client.chat.completions.create(
            model='gpt-4o',  # Use your specific deployment name
            temperature=0.7,
            messages=convo
        )
        content = response.choices[0].message.content
        convo.append({'role': 'assistant', 'content': content})
        return content
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def speak(text):
    # Thai speaker for now (Add functionality to change speakers later)
    speech_config.speech_synthesis_voice_name = "th-TH-PremwadeeNeural"
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config)
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(text))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(
                cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(
                    cancellation_details.error_details))
    except Exception as e:
        print(f"An error occurred during speech synthesis: {str(e)}")


while True:
    prompt = input('USER: ')
    response = get_response(prompt)
    if response:
        print(f'AI: {response}')
        # speak(response)
    else:
        print("Failed to get a response from the AI.")
