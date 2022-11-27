import subprocess
import csv

caseslist = []

for i in range(10, 91, 10):
    print(f"running case {i}")
    size = i
    average_runtime = 0.0
    for j in range(10):
        casename = f"cases/small/case{i}_{j}.csv"
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        average_runtime += float(time)
    caseslist.append((size, average_runtime/10))

for i in range(100, 501, 100):
    print(f"running case {i}")
    size = i
    average_runtime = 0.0
    for j in range(10):
        casename = f"cases/large/case{i}_{j}.csv"
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        average_runtime += float(time)
    caseslist.append((size, average_runtime/10))


print(caseslist)

# output to csv
with open("./metrics/serial_time.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(caseslist)
