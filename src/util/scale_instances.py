import os

path = '../../data/instances/original'
dest_path = '../../data/instances/new'

c = 250

print("Instances:\n", os.listdir(path))

for filename in os.listdir(path):
    with open(os.path.join(path, filename)) as f:
        n = int(f.readline())
        
        vectors = []
        for _ in range(n):
            vector = list(map(float, f.readline().split()))
            vectors.append([c * val for val in vector])
        f.close()
        
        # Write the modified vectors to a new file
        new_filename = os.path.join(dest_path, filename)
        with open(new_filename, "w") as new_file:
            new_file.write(str(n) + "\n")
            for vector in vectors:
                new_file.write(" ".join(map(str, vector)) + "\n")
    