import os
import logging
from datetime import datetime

class Logger():
    def __init__(self, log_dir, log_file):
        self.log_dir = log_dir
        self.log_file = log_file

        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a handler and set the logging level
        handler = logging.FileHandler(os.path.join(self.log_dir, self.log_file))
        handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

        # Clean up old log files
        self.cleanup_old_log_files()

    def cleanup_old_log_files(self):
        if len(os.listdir(self.log_dir)) > 20:
            files = os.listdir(self.log_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            os.remove(os.path.join(self.log_dir, files[0]))

    def log(self, message):
        if type(message) is not str:
            message = str(message)
        self.logger.info(message)
