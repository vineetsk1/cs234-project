import os
import datetime
import sys
from utils import args_to_str

# Simple logger that writes to console and file.
# Maintains a buffer of lines to write out to the file.
class Logger():
    def __init__(self, args, sep=" ", split="\n", directory="logs"):
        self.args = args
        self.fname = args_to_str(args)
        self.lines = []

        self.sep = sep
        self.split = split
        self.dir = directory

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.print_args()


    def print_args(self):
        self.print("Command: python " + " ".join(sys.argv))
        self.print("Running at day/time", str(datetime.datetime.now()))
        self.print("Using settings...")
        vargs = vars(self.args)
        for key in vargs:
            self.print("\t", key, vargs[key])
        self.print("Saving to", self.fname)

    def print(self, *args, **kwargs):
        if kwargs:
            raise TypeError("invalid keyword arguments to print()")

        line = [str(arg) for arg in args]
        line = self.sep.join(line)
        self.lines.append(line)
        print(line)

    def close(self):
        self.print("Saving log...")
        with open(os.path.join(self.dir, self.fname), 'w') as f:
            f.write(self.split.join(self.lines))