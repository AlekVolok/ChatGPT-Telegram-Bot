"""
This bot checks giving youtube link and tries to download it and return a downloaded file
"""
import sys
import re
import os
import subprocess
import pydub

import uuid
import json
import html
import tempfile
import tiktoken
import openai
import asyncio
import logging
import traceback
from pathlib import Path
from typing import NoReturn, Optional, Any
from datetime import datetime

import telegram
from telegram import __version__ as TG_VER
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import Bot, Update
from telegram.error import Forbidden, NetworkError
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_ASSISTANT")

allowed_telegram_usernames = []

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
        "description": "ChatGPT is that well-known model. It's <b>fast</b> and <b>cheap</b>. Ideal for everyday tasks. If there are some tasks it can't handle, try the <b>GPT-4</b>.",
        "price_per_1000_input_tokens": 0.002,
        "price_per_1000_output_tokens": 0.002,
        "scores": {
            "Smart": 3,
            "Fast": 5,
            "Cheap": 5
        }
    },
    "gpt-4-0613": {
        "type": "chat_completion",
        "name": "gpt-4-0613",
        "description": "GPT-4 is the <b>smartest</b> and most advanced model in the world. But it is slower and not as cost-efficient as ChatGPT. Best choice for <b>complex</b> intellectual tasks.",
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
        "price_per_1_min": 0.006
    }
}


import uuid
from datetime import datetime
from typing import Any, Optional
from pathlib import Path


class Database:
    def __init__(self):
        self.data_dir = Path('UserDatabase')
        self.data_dir.mkdir(exist_ok=True)

    def _get_user_file(self, user_id: int):
        return self.data_dir / f"{user_id}.json"

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        user_file = self._get_user_file(user_id)
        if user_file.exists():
            return True
        elif raise_exception:
            raise ValueError(f"User {user_id} does not exist")
        else:
            return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        if not self.check_if_user_exists(user_id):
            user_data = {
                'id': user_id,
                'chat_id': chat_id,
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'last_interaction': datetime.now().isoformat(),
                'first_seen': datetime.now().isoformat(),
                'current_dialog_id': None,
                'current_chat_mode': "assistant",
                'current_model': "gpt-3.5-turbo",
                'dialogs': {},
                'user_tokens': {},
                'dialog_messages': {}
            }

            with self._get_user_file(user_id).open('w') as f:
                json.dump(user_data, f)

    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)
        dialog_id = str(uuid.uuid4())

        with self._get_user_file(user_id).open('r+') as f:
            user_data = json.load(f)
            user_data['dialogs'][dialog_id] = {
                'chat_mode': user_data['current_chat_mode'],
                'start_time': datetime.now().isoformat(),
                'model': user_data['current_model']
            }
            user_data['current_dialog_id'] = dialog_id
            f.seek(0)
            json.dump(user_data, f)

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)

        with self._get_user_file(user_id).open() as f:
            user_data = json.load(f)
            return user_data.get(key)

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)

        with self._get_user_file(user_id).open('r+') as f:
            user_data = json.load(f)
            user_data[key] = value
            f.seek(0)
            json.dump(user_data, f)

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        with self._get_user_file(user_id).open() as f:
            user_data = json.load(f)
            if dialog_id is None:
                dialog_id = user_data['current_dialog_id']
            return user_data['dialog_messages'].get(dialog_id, [])

    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        with self._get_user_file(user_id).open('r+') as f:
            user_data = json.load(f)
            if dialog_id is None:
                dialog_id = user_data['current_dialog_id']
            user_data['dialog_messages'][dialog_id] = dialog_messages
            f.seek(0)
            json.dump(user_data, f)
# DATABASE NEW END

class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in {"gpt-3.5-turbo", "gpt-4-0613"}, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4-0613"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4-0613"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            yield "not_finished", answer

                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                elif self.model == "text-davinci-003":
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r_gen = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=prompt,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        answer += r_item.choices[0].text
                        yield "not_finished", answer

                    n_input_tokens, n_output_tokens = self._count_tokens_from_prompt(prompt, answer, model=self.model)

                answer = self._postprocess_answer(answer)

            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        # prompt = CHAT_MODES[chat_mode]["prompt_start"]
        prompt = ""
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        # prompt = CHAT_MODES[chat_mode]["prompt_start"]
        prompt = ""

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0613":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]

def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


db = Database()
logger = logging.getLogger(__name__)
user_semaphores = {}

async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", "gpt-3.5-turbo")

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with GPT-3.5 OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    reply_text += "\nAnd now... ask me anything!"

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)


async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    async with user_semaphores[user_id]:
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - datetime.fromisoformat(db.get_user_attribute(user_id, "last_interaction"))).seconds > 2000 and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{CHAT_MODES[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            message = message or update.message.text

            current_model = db.get_user_attribute(user_id, "current_model")
            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[CHAT_MODES[chat_mode]["parse_mode"]]

            chatgpt_instance = ChatGPT(model=current_model)
            # if config.enable_message_streaming:
            gen = chatgpt_instance.send_message_stream(message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            # else:
            #     answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
            #         message,
            #         dialog_messages=dialog_messages,
            #         chat_mode=chat_mode
            #     )

                # async def fake_gen():
                #     yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                # gen = fake_gen()

            prev_answer = ""
            async for gen_item in gen:
                status = gen_item[0]
                if status == "not_finished":
                    status, answer = gen_item
                elif status == "finished":
                    status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                else:
                    raise ValueError(f"Streaming status {status} is unknown")

                answer = answer[:4096]  # telegram message limit

                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": message, "bot": answer, "date": datetime.now().isoformat()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            # db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            # await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

    voice = update.message.voice
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)

        # convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")

        # transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await transcribe_audio(f)

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    # db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

    keyboard = []
    for chat_mode, chat_mode_dict in CHAT_MODES.items():
        keyboard.append([InlineKeyboardButton(chat_mode_dict["name"], callback_data=f"set_chat_mode|{chat_mode}")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("Select chat mode:", reply_markup=reply_markup)


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await query.edit_message_text(f"{CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


def get_settings_menu(user_id: int):

    current_model = db.get_user_attribute(user_id, "current_model")
    text = MODELS_INFO[current_model]["description"]

    text += "\n\n"
    score_dict = MODELS_INFO[current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in ["gpt-3.5-turbo", "gpt-4-0613"]:
        title = MODELS_INFO[model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now().isoformat())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def edited_message_handle(update: Update, context: CallbackContext):
    text = "🥲 Unfortunately, message <b>editing</b> is not supported"
    await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/mode", "Select chat mode"),
        BotCommand("/retry", "Re-generate response for previous query"),
        BotCommand("/balance", "Show balance"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/help", "Show help message"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(allowed_telegram_usernames) > 0:
        usernames = [x for x in allowed_telegram_usernames if isinstance(x, str)]
        user_ids = [x for x in allowed_telegram_usernames if isinstance(x, int)]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
