import subprocess
import csv

caseslist = []

for i in range(10, 91, 10):
    print(f"running case {i}")
    size = i
    versions = []
    for j in range(10):
        casename = f"cases/small/case{i}_{j}.csv"
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        iterations = result.stdout.decode('utf-8').split('\r')[2].strip().split(' ')[1]
        versions.append(iterations)
    caseslist.append((size, versions))

for i in range(100, 501, 100):
    print(f"running case {i}")
    size = i
    versions = []
    for j in range(10):
        casename = f"cases/large/case{i}_{j}.csv"
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        iterations = result.stdout.decode('utf-8').split('\r')[2].strip().split(' ')[1]
        versions.append(iterations)
    caseslist.append((size, versions))

print(caseslist)

# output to csv
with open("./metrics/serial_iterations.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(caseslist)
