import logging
import os


class FileLogger:

    def __init__(self, output_dir, is_master=False, is_rank0=False, log_to_file=False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Log to console if rank 0, Log to console and file if master
        if not is_rank0:
            self.logger = NoOp()
        else:
            self.logger = self.get_logger(output_dir, log_to_file=(is_master and log_to_file))


    def get_logger(self, output_dir, log_to_file=True):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        
        if log_to_file:
            vlog = logging.FileHandler(output_dir+'/verbose.log')
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            logger.addHandler(vlog)

            eventlog = logging.FileHandler(output_dir+'/event.log')
            eventlog.setLevel(logging.WARN)
            eventlog.setFormatter(formatter)
            logger.addHandler(eventlog)

            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(formatter)
            logger.addHandler(debuglog)
      
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op
