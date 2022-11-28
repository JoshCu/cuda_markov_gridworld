import subprocess
import csv

caseslist = []


#computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
#total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]

for i in range(10, 91, 10):
    print(f"running case {i}")
    size = i
    average_runtime = 0.0
    for j in range(10):
        casename = f"cases/small/case{i}_{j}.csv"
        result = subprocess.run(['cuda/gputest.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        average_runtime += float(total_time)
    caseslist.append((size, average_runtime/10))

for i in range(100, 501, 100):
    print(f"running case {i}")
    size = i
    average_runtime = 0.0
    for j in range(10):
        casename = f"cases/large/case{i}_{j}.csv"
        result = subprocess.run(['cuda/gputest.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        average_runtime += float(total_time)
    caseslist.append((size, average_runtime/10))


# output to csv
with open("./metrics/parallel_time.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(caseslist)
