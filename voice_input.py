import speech_recognition as sr
import sys

recognizer = sr.Recognizer()

# Dictionary of supported languages with their codes
LANGUAGES = {
    "english": "en-US",
    "french": "fr-FR",
    "spanish": "es-ES",
    "german": "de-DE",
    "italian": "it-IT",
    "japanese": "ja-JP",
    "korean": "ko-KR",
    "chinese (mandarin)": "zh-CN",
    "thai": "th-TH",
    "cantonese": "yue-Hant-HK"
}


def select_language():
    print("Available languages:")
    for i, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"{i}. {lang.capitalize()}")

    while True:
        try:
            choice = int(input("Select a language number: "))
            if 1 <= choice <= len(LANGUAGES):
                return list(LANGUAGES.values())[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


print("Press Ctrl+C at any time to exit the program.")

# Select language
language = select_language()
print(
    f"Selected language: {list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)].capitalize()}")

while True:
    try:
        input("Press Enter to start listening...")

        with sr.Microphone() as mic:
            print("Adjusting for ambient noise. Please wait...")
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            print("Listening... Speak now!")
            audio = recognizer.listen(
                mic, timeout=None, phrase_time_limit=None)

        print("Processing...")
        text = recognizer.recognize_google(audio, language=language)
        text = text.lower()

        print(f"Recognized: {text}")

    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(
            f"Could not request results from Speech Recognition service; {e}")
    except KeyboardInterrupt:
        print("\nExiting the program.")
        sys.exit(0)

    print("\nReady for next input...")
