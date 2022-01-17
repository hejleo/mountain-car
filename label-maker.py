import glob, os
import csv

os.chdir("/home/pulp/openai-gym/dataset/test")
f = open('/home/pulp/openai-gym/dataset/test-labels', 'w')
writer = csv.writer(f)


for file in glob.glob("*.jpg"):
    print(file[:-4])
    if file[0]=='l':
        writer.writerow([file[:-4]]+[0])
    else:
        writer.writerow([file[:-4]]+ [1])

f.close()
