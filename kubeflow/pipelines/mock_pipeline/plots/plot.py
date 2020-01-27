import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("get_data.log")])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.warning(f"Parsed unknown args: {unknown}")
    kwargs = dict(vars(args))

    try:
        logger.info('Plots...')
        # TODO: plots
        logger.info('Done.')
    except Exception as e:
        logger.exception(e)
