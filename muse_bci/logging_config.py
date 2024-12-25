import logging
import yaml

def setup_logging(log_name, level):
    # Root logger configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{log_name}.log')
        ]
    )

def setup_logging_from_yaml():
    with open('logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)