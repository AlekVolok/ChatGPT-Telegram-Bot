import os

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
TEXT_EMBEDDING_CHUNK_SIZE=300
VECTOR_FIELD_NAME="content_vector"
PREFIX = "sportsdoc"  
INDEX_NAME = "f1-index"
OPENAI_API = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_ASSISTANT")

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help
"""

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

CHAT_MODES = {
    "assistant" : {
        "name" : "👩🏼‍🎓",
        "welcome_message" : "👩🏼‍🎓 Hi, I'm <b>General Assistant</b>. How can I help you?",
        "promt_start" : "I'm a general assistant",
        "parse_mode" : "html"
    },
    "code_assistant" : {
        "name" : "👩🏼‍🎓 Code Assistant",
        "welcome_message" : "👩🏼‍🎓 Hi, I'm <b>General Assistant</b>. How can I help you?",
        "promt_start" : "I'm code assistant. How can I help you?",
        "parse_mode" : "markdown"
    },
    "text_improver" : {
        "name" : "📝 Text Improver",
        "welcome_message" : "Hi, I'm text improver",
        "promt_start" : "I'm code assistant. How can I help you?",
        "parse_mode" : "html"
    },
    "movie_expert" : {
        "name" : "🎬 Movie Expert",
        "welcome_message" : "Hi, I'm a movie expert",
        "promt_start" : "I'm a movie expert. How can I help you?",
        "parse_mode" : "html"
    }
}

MODELS_INFO = {
    "gpt-3.5-turbo": {
        "type": "chat_completion",
        "name": "ChatGPT",
        "description": "ChatGPT",
        "price_per_1000_input_tokens": 0.002,
        "price_per_1000_output_tokens": 0.002,
        "scores": {
            "Smart": 3,
            "Fast": 5,
            "Cheap": 5
        }
    },
    "gpt-4": {
        "type": "chat_completion",
        "name": "gpt-4",
        "description": "GPT-4",
        "price_per_1000_input_tokens": 0.03,
        "price_per_1000_output_tokens": 0.06,
        "scores": {
            "Smart": 5,
            "Fast": 2,
            "Cheap": 2
        }
    },
    "whisper": {
        "type": "audio",
        "name": "whisper",
        "price_per_1_min": 0.006
    }
}