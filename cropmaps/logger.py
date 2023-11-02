import logging
import coloredlogs

def setup(name = None, level = logging.INFO, propag = False):
    """Logger configuration for the project.
    """
    # Name of each module (_name_)
    logger = logging.getLogger(name)

    # Don't see messages from contained module
    logger.propagate = propag

    # Set module level
    logger.setLevel(level)
    
    # Create a console handler
    console_handler = logging.StreamHandler()

    # Set console_handler level
    console_handler.setLevel(logging.DEBUG)

    # Create colored formatter and add it to the console handler
    console_formatter = coloredlogs.ColoredFormatter('%(asctime)s - %(lineno)s - %(levelname)s - %(name)s - %(message)s', datefmt='%I:%M')
    console_handler.setFormatter(console_formatter)

    # # Create simple formatter and add it to the file handler
    # file_formatter = logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s - %(name)s - %(message)s', datefmt='%I:%M')

    # Add handlers to console and log file
    logger.addHandler(console_handler)

    coloredlogs.install()

    return logger