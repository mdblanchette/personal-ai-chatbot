from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",  # Double-check this URL
    api_version="2023-05-15",  # Use a current API version
    api_key=""
)


def get_response(message, instruction):
    try:
        response = client.chat.completions.create(
            model='gpt-4o',  # Use your specific deployment name
            temperature=1,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": message}
            ]
        )
        print(response.usage)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


result = get_response(
    "Please introduce yourself in Thai and Mandarin.", "You are a professional language tutor.")
if result:
    print(result)
