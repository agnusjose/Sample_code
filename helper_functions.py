import random
import openai
import langchain
from index_functions import load_data
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader

# Define your 'thanks' and 'initial messages' phrases before using them
thanks_phrases = ["Thank you!", "Thanks a lot!", "I appreciate it!", "Many thanks!"]
initial_message_phrases = ["Hello! How can I assist you today?", "Hi! How can I help?", "Hey! What's on your mind?"]

# Main function to generate responses from OpenAI's API, not considering indexed data
def generate_response(prompt, history, model_name, temperature):
    try:
        # Ensure there's enough history to avoid index errors
        if len(history) < 2:
            raise ValueError("Conversation history is too short.")
        
        # Fetching the last message sent by the chatbot from the conversation history
        chatbot_message = history[-1]['content']

        # Fetching the first message that the user sent from the conversation history
        first_message = history[1]['content']

        # Fetching the last message that the user sent from the conversation history
        last_user_message = history[-1]['content']

        # Constructing a comprehensive prompt to feed to OpenAI for generating a response
        full_prompt = f"{prompt}\n\
        ### The original message: {first_message}. \n\
        ### Your latest message to me: {chatbot_message}. \n\
        ### Previous conversation history for context: {history}"

        # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
        api_response = openai.ChatCompletion.create(
            model=model_name,  # The specific OpenAI model to use for generating the response
            temperature=temperature,  # The 'creativity' setting for the response
            messages=[  # The list of message objects to provide conversation history context
                {"role": "system", "content": full_prompt},  # System message to provide instruction
                {"role": "user", "content": last_user_message}  # The last user message to generate a relevant reply
            ]
        )
        
        # Extracting the generated response content from the API response object
        full_response = api_response['choices'][0]['message']['content']

        # Yielding a response object containing the type and content of the generated message
        return {"type": "response", "content": full_response}
    
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return {"type": "error", "content": "Sorry, there was an error generating the response."}
    except ValueError as e:
        print(f"History Error: {e}")
        return {"type": "error", "content": "Sorry, the conversation history is not long enough to generate a response."}
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return {"type": "error", "content": "An unexpected error occurred."}


# Similar to generate_response but also includes indexed data to provide more context-aware and data-driven responses
def generate_response_index(prompt, history, model_name, temperature, chat_engine):
    try:
        # Ensure there's enough history to avoid index errors
        if len(history) < 2:
            raise ValueError("Conversation history is too short.")
        
        # Fetching the last message sent by the chatbot from the conversation history
        chatbot_message = history[-1]['content']

        # Fetching the first message that the user sent from the conversation history
        first_message = history[1]['content']

        # Fetching the last message that the user sent from the conversation history
        last_user_message = history[-1]['content']

        # Constructing a comprehensive prompt to feed to OpenAI for generating a response
        full_prompt = f"{prompt}\n\
        ### The original message: {first_message}. \n\
        ### Your latest message to me: {chatbot_message}. \n\
        ### Previous conversation history for context: {history}"

        # Initializing a variable to store indexed data relevant to the user's last message
        index_response = ""

        # Fetching relevant indexed data based on the last user message using the chat engine
        response = chat_engine.chat(last_user_message)
        
        # Storing the fetched indexed data in a variable
        index_response = response.response

        # Adding the indexed data to the prompt to make the chatbot response more context-aware and data-driven
        full_prompt += f"\n### Relevant data from documents: {index_response}"

        # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
        api_response = openai.ChatCompletion.create(
            model=model_name,  # The specific OpenAI model to use for generating the response
            temperature=temperature,  # The 'creativity' setting for the response
            messages=[  # The list of message objects to provide conversation history context
                {"role": "system", "content": full_prompt},  # System message to provide instruction
                {"role": "user", "content": last_user_message}  # The last user message to generate a relevant reply
            ]
        )
        
        # Extracting the generated response content from the API response object
        full_response = api_response['choices'][0]['message']['content']
        
        # Yielding a response object containing the type and content of the generated message
        return {"type": "response", "content": full_response}
    
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return {"type": "error", "content": "Sorry, there was an error generating the response."}
    except ValueError as e:
        print(f"History Error: {e}")
        return {"type": "error", "content": "Sorry, the conversation history is not long enough to generate a response."}
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return {"type": "error", "content": "An unexpected error occurred."}

#################################################################################
# Additional, specific functions for inspiration:

# Function returns a random thanks phrase to be used as part of the CoPilots reply
def get_thanks_phrase():
    selected_phrase = random.choice(thanks_phrases)
    return selected_phrase

# Function to randomize initial message of CoPilot
def get_initial_message():
    initial_message = random.choice(initial_message_phrases)
    return initial_message

# Function to generate the summary; used in part of the response
def generate_summary(model_name, temperature, summary_prompt):
    try:
        summary_response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert at summarizing information effectively and making others feel understood"},
                {"role": "user", "content": summary_prompt},
            ]
        )
        summary = summary_response['choices'][0]['message']['content']
        print(f"summary: {summary}, model name: {model_name}, temperature: {temperature})")
        return summary
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return "Sorry, I couldn't generate a summary at the moment."

# Function used to enable 'summary' mode in which the CoPilot only responds with bullet points rather than paragraphs
def transform_bullets(content):
    try:
        prompt = f"Summarize the following content in 3 brief bullet points while retaining the structure and conversational tone (using wording like 'you' and 'your idea'):\n{content}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=.2,
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error in transform_bullets: {e}")
        return content  # Return the original content as a fallback

# Function to add relevant stage specific context into prompt
def get_stage_prompt(stage):
    # Implementation dependent on your chatbot's context
    return

# Function to grade the response based on length, relevancy, and depth of response
def grade_response(user_input, assistant_message, idea):
    # Implementation dependent on your chatbot's context
    return       

# Function used to generate a final 'report' at the end of the conversation, summarizing the convo and providing a final recommendation
def generate_final_report():
    # Implementation dependent on your chatbot's context
    return
