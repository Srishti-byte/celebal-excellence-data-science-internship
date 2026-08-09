from src.logger import DriveWiseLogger


def main():
    print("\nInitializing logger...")

    logger = DriveWiseLogger()

    print(
        f"Log File: {logger.log_file}"
    )

    log_file = logger.log_file

    # Preserve existing application logs.
    if log_file.exists():
        original_content = log_file.read_text(
            encoding="utf-8"
        )
    else:
        original_content = None

    try:
        print("\nTesting query logging...")

        logger.log_query(
            "Does Toyota Urban Cruiser Hyryder have ADAS?"
        )

        print("Testing response time logging...")

        logger.log_response_time(
            1.2845
        )

        print("Testing failed query logging...")

        logger.log_failed_query(
            "Test retrieval failure"
        )

        print("Testing retrieval results logging...")

        logger.log_retrieval_results(
            "3 documents retrieved after reranking"
        )

        print("Testing answer generation logging...")

        logger.log_generation_status(
            "success"
        )

        print("\nChecking log file...")

        assert log_file.exists()

        log_content = log_file.read_text(
            encoding="utf-8"
        )

        print("\nLog File Content")
        print("=" * 80)
        print(log_content)

        print("Validation")
        print("=" * 80)

        assert "QUERY" in log_content
        assert "RESPONSE_TIME" in log_content
        assert "FAILED_QUERY" in log_content
        assert "RETRIEVAL_RESULTS" in log_content
        assert "ANSWER_GENERATION" in log_content

        print(
            "All logger validation checks passed."
        )

    finally:
        # Restore the original application log.
        if original_content is None:
            if log_file.exists():
                log_file.unlink()
        else:
            log_file.write_text(
                original_content,
                encoding="utf-8"
            )


if __name__ == "__main__":
    main()