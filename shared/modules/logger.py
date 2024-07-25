import logging


class Logger(logging.Logger):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Logger, cls).__new__(cls)
        return cls.instance

    def __init__(s, output=None, name=None, level=logging.DEBUG):
        super().__init__(name=name, level=level)
        slogger = logging.StreamHandler()
        slogger.setFormatter(s.formatter)
        s.addHandler(slogger)
        if output:
            s.add_file_handler(output)

    def add_file_handler(cls, path):
        flogger = logging.FileHandler(path, encoding="UTF-8")
        flogger.setFormatter(cls.formatter)
        cls.instance.addHandler(flogger)


# to be refactored once more than 1 channel is necessary. Cannot be Singleton there, rather cached
log = logger = Logger()
info = logger.info
warning = logger.warning
error = logger.error


if __name__ == '__main__':
    # logger = Logger()
    # logger2 = Logger()
    # logger.info('asdfsd')
    # logger.error('asdfsd')
    # logger.warning('asdfsd')
    # logger2.debug('asdfsd')
    # logger.add_file_handler(r'C:\Users\seb\tl.txt')
    # logger.info('write')
    # logger2.info('writelalal')
    pass
