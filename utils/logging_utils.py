# utils/logging_utils.py

import logging
import csv
import os

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def log_interaction(session_id, user_query, response, log, score, judgment):
    log_file = 'data/training_data.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['session_id', 'user_query', 'response', 'log', 'score', 'judgment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'session_id': session_id, 'user_query': user_query, 'response': response, 'log': log, 'score': score, 'judgment': judgment})
