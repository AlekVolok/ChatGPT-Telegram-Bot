from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import uuid
import json

class ChatGptDatabase:
    """
    This class aims to manage a local user database for a chatbot, presumably based on a GPT model. 
    The data is stored in individual JSON files, one per user, identified by the user's ID.
    """
    def __init__(self):
        """
        Initialize the ChatGptDatabase by setting the data directory path.
        """
        self.data_dir = Path("user_database")
        self.data_dir.mkdir(exist_ok=True)

    def _get_user_file(self, user_id: int):
        """
        Get the file path for a given user ID.

        :param user_id: User's identifier.
        :return: Path object to the user's data file.
        """
        return self.data_dir / f"{user_id}.json"

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        """
        Check if a user's data file exists.

        :param user_id: User's identifier.
        :param raise_exception: If True, raises an exception if the user doesn't exist.
        :return: True if user file exists, False otherwise.
        """
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