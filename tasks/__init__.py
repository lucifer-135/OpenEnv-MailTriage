from tasks.task_easy import TASK_NAME as task1
from tasks.task_medium import TASK_NAME as task2
from tasks.task_hard import TASK_NAME as task3

TASK_REGISTRY = {
    task1: "easy",
    task2: "medium",
    task3: "hard"
}

__all__ = ["TASK_REGISTRY"]
