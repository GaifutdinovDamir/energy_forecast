import logging
import os
from datetime import datetime


def setup_logging(
    log_dir: str = "logs", log_level=logging.INFO, task_name: str = "pipeline"
) -> None:
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{task_name}_{timestamp}.log")

    # Основной формат логов
    log_format = "%(asctime)s — %(levelname)s — %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Настройка логгера
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    logging.info("🔧 Logging initialized")
    logging.info(f"📁 Logs are saved to: {log_file}")
