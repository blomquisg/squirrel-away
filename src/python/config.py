import os
import configparser
import logging
import sys
import contextlib

class Config:
    def __init__(self, env=None, config_file="config/config.ini"):
        if env is None:
            env = os.getenv("ENVIRONMENT", "dev")

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.env = env
        self._load_settings()
        self._create_logger()

    @contextlib.contextmanager
    def disable_logging(self, level=logging.CRITICAL):
        """
        Context Manager that temporarily disables logging
        """
        previous_level = self.logger.getEffectiveLevel()
        try:
            yield
        finally:
            self.logger.setLevel(previous_level)

    def _load_settings(self):
        self.data_dir = self._get_config("data_dir")
        self.experiments_dir = self._get_config("experiments_dir")
        self.log_level = self._get_config("log_level")
        self.log_stdout = self._get_config("log_stdout")    

    def _get_config(self, config_name):
        return self.config.get(self.env, config_name, fallback=self.config.get("default", config_name))
    
    def _create_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        if self.log_stdout and not self.logger.hasHandlers():
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(self.log_level)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            stdout_handler.setFormatter(formatter)
            self.logger.addHandler(stdout_handler)

    def __repr__(self):
        _configs  = f"env={self.env}, "
        _configs += f"data_dir={self.data_dir}, "
        _configs += f"experiments_dir={self.experiments_dir}, "
        _configs += f"log_level={self.log_level}, "

        return f"Config({_configs})"
    
if __name__ == "__main__":
    env = os.getenv("ENVIRONMENT", "dev")
    config = Config(env=env)

    config.logger.debug("Hello!")

    print(config)

    
