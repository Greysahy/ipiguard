user_task_num = {
    "banking": 16,
    "workspace": 40,
    "slack": 21,
    "travel": 20,
}

injection_task_num = {
    "banking": 9,
    "workspace": 6,
    "slack": 5,
    "travel": 7,
}

def initialize_dataset(suite_name, benign=False):
    delta = 0
    if suite_name == 'slack':
        delta = 1

    if benign:
        return [
            i for i in range(user_task_num[suite_name])
        ]
    return [
        (i, j + delta) for i in range(user_task_num[suite_name]) for j in range(injection_task_num[suite_name])
    ]
