from enum import Enum


class LogLevel(Enum):
    Debug = 1
    Info = 2
    Warning = 3
    Error = 4
    Critical = 5
    

class Logger:
    def __init__(self, log_level=LogLevel.Info):
        self.log_level = log_level

    def set_log_level(self, log_level):
        self.log_level = log_level
    
    def log(self, message, level=LogLevel.Info):
        if level.value >= self.log_level.value:
            print(f"[{level.name}] {message}", flush=True)

    def debug(self, message):
        self.log(message, level=LogLevel.Debug)

    def info(self, message):
        self.log(message, level=LogLevel.Info)

    def warning(self, message):
        self.log(message, level=LogLevel.Warning)

    def error(self, message):
        self.log(message, level=LogLevel.Error)
    
    def critical(self, message):
        self.log(message, level=LogLevel.Critical)

log = Logger()  # singleton
