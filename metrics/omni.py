import subprocess
import csv


runtime_caseslist = []
computation_caselist = []

for i in range(10, 91, 10):
    print(f"running case {i}")
    size = i
    average_runtime_cpu = 0.0
    average_runtime_gpu4 = 0.0
    average_runtime_gpu8 = 0.0
    average_runtime_gpu16 = 0.0
    average_runtime_gpu32 = 0.0
    average_computation_time_cpu = 0.0
    average_computation_time_gpu4 = 0.0
    average_computation_time_gpu8 = 0.0
    average_computation_time_gpu16 = 0.0
    average_computation_time_gpu32 = 0.0
    for j in range(10):
        casename = f"cases/small/case{i}_{j}.csv"
        # CPU
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_cpu += float(computation_time)
        average_runtime_cpu += float(time)
        # GPU
        # 4
        result = subprocess.run(['cuda/gputest4.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu4 += float(computation_time)
        average_runtime_gpu4 += float(total_time)
        # 8
        result = subprocess.run(['cuda/gputest8.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu8 += float(computation_time)
        average_runtime_gpu8 += float(total_time)
        # 16
        result = subprocess.run(['cuda/gputest16.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu16 += float(computation_time)
        average_runtime_gpu16 += float(total_time)
        # 32
        result = subprocess.run(['cuda/gputest32.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu32 += float(computation_time)
        average_runtime_gpu32 += float(total_time)
    runtime_caseslist.append((size, average_runtime_cpu/10, average_runtime_gpu4/10,
                             average_runtime_gpu8/10, average_runtime_gpu16/10, average_runtime_gpu32/10))
    computation_caselist.append(
        (size, average_computation_time_cpu / 10, average_computation_time_gpu4 / 10, average_computation_time_gpu8 /
         10, average_computation_time_gpu16 / 10, average_computation_time_gpu32 / 10))

for i in range(100, 501, 100):
    print(f"running case {i}")
    size = i
    average_runtime_cpu = 0.0
    average_runtime_gpu4 = 0.0
    average_runtime_gpu8 = 0.0
    average_runtime_gpu16 = 0.0
    average_runtime_gpu32 = 0.0
    average_computation_time_cpu = 0.0
    average_computation_time_gpu4 = 0.0
    average_computation_time_gpu8 = 0.0
    average_computation_time_gpu16 = 0.0
    average_computation_time_gpu32 = 0.0
    for j in range(10):
        casename = f"cases/large/case{i}_{j}.csv"
        # CPU
        result = subprocess.run(['c_serial/test.exe', casename], stdout=subprocess.PIPE)
        time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_cpu += float(computation_time)
        average_runtime_cpu += float(time)
        # GPU
        # 4
        result = subprocess.run(['cuda/gputest4.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu4 += float(computation_time)
        average_runtime_gpu4 += float(total_time)
        # 8
        result = subprocess.run(['cuda/gputest8.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu8 += float(computation_time)
        average_runtime_gpu8 += float(total_time)
        # 16
        result = subprocess.run(['cuda/gputest16.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu16 += float(computation_time)
        average_runtime_gpu16 += float(total_time)
        # 32
        result = subprocess.run(['cuda/gputest32.exe', casename], stdout=subprocess.PIPE)
        total_time = result.stdout.decode('utf-8').split('\r')[5].strip().split(' ')[-1]
        computation_time = result.stdout.decode('utf-8').split('\r')[4].strip().split(' ')[-1]
        average_computation_time_gpu32 += float(computation_time)
        average_runtime_gpu32 += float(total_time)
    runtime_caseslist.append((size, average_runtime_cpu/10, average_runtime_gpu4/10,
                             average_runtime_gpu8/10, average_runtime_gpu16/10, average_runtime_gpu32/10))
    computation_caselist.append(
        (size, average_computation_time_cpu / 10, average_computation_time_gpu4 / 10, average_computation_time_gpu8 /
         10, average_computation_time_gpu16 / 10, average_computation_time_gpu32 / 10))


# output to csv
with open("./metrics/runtime.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(runtime_caseslist)

# output to csv
with open("./metrics/computation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(computation_caselist)
