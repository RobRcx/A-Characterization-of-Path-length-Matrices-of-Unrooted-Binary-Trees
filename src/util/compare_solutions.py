import os
import filecmp

#path1 = "./original"
path1 = "../../data/solutions/original"
path2 = "../../data/solutions/new/BMEP_classic"
instance_path = "../../data/instances/original"

print(os.listdir(path1))

for f in os.listdir(path1):
    filepath1 = os.path.join(path1, f)
    filepath2 = os.path.join(path2, f)
    # print("Comparing ", filepath1, "and", filepath2)
    if not os.path.isfile(filepath2):
        continue
    if not filecmp.cmp(filepath1, filepath2):
        print(filepath1, "and", filepath2, " differ!!!")
    splits = f.split('_')
    instance_name = splits[0] + "_" + splits[1] + ".txt"
    print(instance_name)
    print(f, "OK")