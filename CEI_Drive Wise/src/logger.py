import logging

from src.config import LOGS_PATH


class DriveWiseLogger:
    def __init__(self):
        LOGS_PATH.mkdir(parents=True, exist_ok=True)

        self.log_file = LOGS_PATH / "drive_wise.log"

        self.logger = logging.getLogger("drive_wise")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(
                self.log_file,
                encoding="utf-8"
            )

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_query(self, query):
        self.logger.info(
            f"QUERY | {query}"
        )

    def log_response_time(self, response_time):
        self.logger.info(
            f"RESPONSE_TIME | {response_time:.4f}s"
        )

    def log_failed_query(self, error):
        self.logger.error(
            f"FAILED_QUERY | {error}"
        )

    def log_retrieval_results(self, results):
        self.logger.info(
            f"RETRIEVAL_RESULTS | {results}"
        )

    def log_generation_status(self, status):
        self.logger.info(
            f"ANSWER_GENERATION | status={status}"
        )

    def log_evaluation_metrics(self, metrics):
        self.logger.info(
            f"EVALUATION_METRICS | {metrics}"
        )

    def log_request(
        self,
        query,
        response_time,
        retrieval_results,
        generation_status
    ):
        self.log_query(query)
        self.log_response_time(response_time)
        self.log_retrieval_results(retrieval_results)
        self.log_generation_status(generation_status)