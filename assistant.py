from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SERVICE_REGION = os.getenv('AZURE_SERVICE_REGION')

LANGUAGE_VOICE_MAPPING = {
    'english': 'en-US-AvaMultilingualNeural',
    'thai': 'th-TH-PremwadeeNeural',
    'mandarin': 'zh-CN-XiaoxiaoNeural',
    'cantonese': 'yue-CN-XiaoMinNeural',
}

LANGUAGE_CODE_MAPPING = {
    'english': 'en-US',
    'thai': 'th-TH',
    'mandarin': 'zh-CN',
    'cantonese': 'yue-CN',
}

current_language = 'english'  # Default language

recognizer = sr.Recognizer()

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
    '15. Your default language at the beginning of the conversation is English. The user may request a different language at any time.\n'
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


def speak(text, language):
    if language.lower() in LANGUAGE_VOICE_MAPPING:
        speech_config.speech_synthesis_voice_name = LANGUAGE_VOICE_MAPPING[language.lower(
        )]
    else:
        print(
            f"Warning: No voice model found for {language}. Using default voice.")

    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config)
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        # if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        #     print("Speech synthesized for text [{}]".format(text))
        # elif result.reason == speechsdk.ResultReason.Canceled:
        #     cancellation_details = result.cancellation_details
        #     print("Speech synthesis canceled: {}".format(
        #         cancellation_details.reason))
        #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
        #         print("Error details: {}".format(
        #             cancellation_details.error_details))
    except Exception as e:
        print(f"An error occurred during speech synthesis: {str(e)}")


def get_voice_input():
    global current_language
    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Listening... (Current language: {current_language})")
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

    try:
        text = recognizer.recognize_google(
            audio, language=LANGUAGE_CODE_MAPPING.get(current_language, 'en-US'))
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError as e:
        print(
            f"Could not request results from Speech Recognition service; {e}")
        return None


def change_language(new_language):
    global current_language
    if new_language in LANGUAGE_VOICE_MAPPING:
        current_language = new_language
        language_instruction = f"From now on, please respond in {new_language}. This is a language change instruction."
        convo.append({'role': 'system', 'content': language_instruction})
        return f"Language changed to {current_language}. The AI will now respond in {current_language}."
    else:
        return f"Sorry, {new_language} is not supported. Please try another language."


print("Welcome to the AI Language Tutor!")
print("You can type 'voice' to switch to voice input mode.")
print("To change the language, type or say 'change language to [language]'.")

while True:
    print("\nChoose input method:")
    print("1. Type your message")
    print("2. Use voice input")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        prompt = input('USER: ')
    elif choice == '2':
        prompt = get_voice_input()
        if prompt is None:
            continue
    else:
        print("Invalid choice. Please enter 1 or 2.")
        continue

    # Check if the user wants to change the language
    if prompt.lower().startswith('change language to '):
        new_language = prompt[19:].strip().lower()
        response = change_language(new_language)
        print(response)
        continue

    response = get_response(prompt)
    if response:
        print(f'AI: {response}')
        speak(response, current_language)
    else:
        print("Failed to get a response from the AI.")
