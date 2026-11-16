import argparse
import asyncio
from utils.configuration import CONFIG, read_config
from utils.experiments import Queue


def main():
    parser = argparse.ArgumentParser(description='Run selected experiments sequentially, continuing after failures.')
    parser.add_argument('batch', nargs='?', default='batch_config.yaml')
    parser.add_argument('--config', default=str(CONFIG))
    args = parser.parse_args()
    queue = Queue()

    async def run():
        try:
            return await queue.experiments(read_config(args.batch), config_path=args.config)
        finally:
            await queue.cancel()
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        parser.exit(2, f'batch: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
