class Logger(object):
    def __init__(self, type : str = "cmd", file_path : str = None):
        try:
            self.file = open(file_path, "w")
        except Exception as e:
            print(f"ERROR: Cannot create report file {file_path}")
            print(f"Reason: {e}")
            self.file = None

        self.change_type(type)

    def change_type(self, type : str = "cmd"):
        if type == "cmd":
            self.type = 0
        elif type == "txt":
            if self.file is None:
                print(f"ERROR: log file is None. Impossible to switch to text file logging. Logging mode stay on 'cmd'")
                self.type = 0
            else:
                self.type = 1
        else:
            print(f"ERROR: Unknown log type. Use 'cmd' or 'txt. Will use 'cmd' as default")
            self.type = 0

    def log(self, message : str):
        if self.type == 0:
            print(message)
        elif self.type == 1:
            self.file.write(message + "\n")