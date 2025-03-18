import os
import math

files = os.listdir()
print(files)
tmp = "tmp"
for file in files:
    try:
        src = open(file, "r")
    except Exception as ex:
        print(ex)
        continue
    spl = file.split(".")
    if len(spl) != 2 or (len(spl) == 2 and file.split(".")[1] != "txt"):
        continue
    print(f"Processing {file}...")
    lines = src.readlines()
    src.close()
    dest = open(tmp, "w+")
    for l in lines[0:3]:
        dest.write(l)
    for l in lines[3:]:
        v = l.split(" ")
        for x in v:
            if x == "\n":
                dest.write(x)
            else:
                dest.write(f"{round(float(x))} ")
    dest.close()
    # RENAME FILES
    os.remove(file)
    os.rename(tmp, file)

