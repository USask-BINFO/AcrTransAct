import os
from datetime import datetime
class Logger():
    def __init__(self, log_dir, log_file):
        self.log_dir = log_dir
        self.log_file = log_file

        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
            
        # if there are more than 10 files in the log folder delete the oldest ones
        if len(os.listdir(self.log_dir)) > 20:
            files = os.listdir(self.log_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            os.remove(os.path.join(self.log_dir, files[0]))

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M")
        if type(message) is not str:
            message = str(message)
        with open(os.path.join(self.log_dir, self.log_file), 'a') as f:
            f.write(f"[TIME: {timestamp}] "+ message + '\n')