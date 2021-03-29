import os

directory = 'D:\\projects\\493_GAN\\RenyiGan-TensorFlow2\\data\\renyiganV_2\\AlphaG=9_AlphaD=0.5\\'
avglist = []
max = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith("y.txt"):
            pa = os.path.join(root, file)
            with open(pa, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                split = last_line.split(" ")
                fid = float(split[3])
                avglist.append(fid)
                if fid > max:
                    max = fid
            with open(directory + 'data.txt', "a") as file_object:
                file_object.write(last_line + '\n')
avg = sum(avglist) / len(avglist)
with open(directory + 'data.txt', "a") as file_object:
    file_object.write(' Avg FID: ' + str(avg))
    file_object.write(' Max FID: ' + str(max))


