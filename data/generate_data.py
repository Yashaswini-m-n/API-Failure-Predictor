import pandas as pd
import random

data = []

for _ in range(5000):

    response_time = random.randint(50, 2500)  # ms
    status_code = random.choices(
        [200, 200, 200, 200, 500, 503], weights=[70,70,70,70,10,10]
    )[0]

    cpu = random.randint(10, 100)
    memory = random.randint(10, 100)

    failure = 0

    if response_time > random.randint(1300, 1700):
        failure = 1
    if status_code != 200:
        failure = 1
    if cpu > 85 or memory > 90:
        failure = 1

    data.append([
        response_time,
        status_code,
        cpu,
        memory,
        failure
    ])

df = pd.DataFrame(data, columns=[
    "response_time",
    "status_code",
    "cpu",
    "memory",
    "failure"
])

df.to_csv("data/system_logs.csv", index=False)

print("Dataset generated successfully ✅")