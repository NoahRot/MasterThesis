from typing import Union

class Logger(object):
    """
    Handle the logging. Can log data in CMD and text files

    Parameters
    ----------
    type : str, default="cmd"
        Type of log. Can be either "cmd" or "txt". If another type is given,
        it will use the "cmd" mode.
    file_path : str or None, default=None
        Path to the file for text file logging mode. If text mode is enable, but
        file_path is None, it will automatically switch to "cmd" mode.
    """
    
    def __init__(self, type : str = "cmd", file_path : Union[str, None] = None):
        try:
            self.file = open(file_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"ERROR: Cannot create report file {file_path}")
            print(f"Reason: {e}")
            self.file = None

        self.change_type(type)

    def change_type(self, type : str = "cmd"):
        """
        Change type of logging mode

        Parameters
        ----------
        type : str, default="cmd"
            Type of log. Can be either "cmd" or "txt". If another type is given
            or no file path has been provided, it will use the "cmd" mode.
        """
        
        if type == "cmd":
            self.type = 0

        elif type == "txt":
            # Check if fiel exist
            if self.file is None:
                print(f"WARNING: log file is None. Impossible to switch to text file logging. Logging mode stay on 'cmd'")
                self.type = 0
            else:
                self.type = 1
        
        else:
            # Nonexistant logging mode
            print(f"WARNING: Unknown log type. Use 'cmd' or 'txt. Will use 'cmd' as default")
            self.type = 0

    def log(self, message : str):
        """
        Log a message

        Parameters
        ----------
        message : str
            Message to log
        """
        
        if self.type == 0:
            print(message)
        elif self.type == 1:
            self.file.write(message + "\n")