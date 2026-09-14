
from pathlib import Path

from Voice_Cloning.config import TRAINED_MODELS_DIRECTORY


class ModelRegistry:
    """
    Keeps track of fine-tuned voice models for users.
    """

    def __init__(self):
        self.models_directory = Path(
            TRAINED_MODELS_DIRECTORY
        )

    def get_model_path(self, user_id):
        """
        Return the fine-tuned model path for a user.
        """

        user_directory = (
            self.models_directory / user_id
        )

        if not user_directory.exists():
            return None

        model_files = list(
            user_directory.glob("*.pth")
        )

        if not model_files:
            model_files = list(
                user_directory.glob("*.pt")
            )

        if not model_files:
            return None

        return model_files[0]

    def has_model(self, user_id):
        """
        Check whether a fine-tuned model exists
        for the given user.
        """

        return self.get_model_path(user_id) is not None
