from pathlib import Path
from os import walk

cwd = str(Path().resolve())
mypath = cwd + '/' + str(1606820400) + '/' + str(10000000)
filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
print(filenames)