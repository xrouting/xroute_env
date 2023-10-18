import multiprocessing
import re
from typing import List, Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..utils.fastapi import resp_success, resp_fail
from ..utils.openroad import OpenRoadTask

router = APIRouter()

openroad_task_list = {}

regex_ip_port = r'^(\d{1,3}\.){3}\d{1,3}:\d{4,5}$'


class Task(BaseModel):
    testcase_name: str
    task_mode: str = 'normal'
    custom_strategies: List[List[Any]] = []
    parallel_workers: int = multiprocessing.cpu_count()
    api_addresses: List[str] = []
    api_timeout: int = 30000
    net_ordering_evaluation_mode: int = 2
    droute_end_iter: int = -1


@router.get("/")
async def get_tasks():
    return resp_success(data=list(openroad_task_list.keys()))


@router.post("/")
async def create_task(task: Task):
    try:
        # parameter validation
        if task.task_mode in ['training', 'evaluation']:
            if task.api_addresses is None or len(task.api_addresses) == 0:
                raise ValueError('api_addresses is required in training or evaluation mode')
            for addr in task.api_addresses:
                if re.match(regex_ip_port, addr) is None:
                    raise ValueError(f'invalid api_address: {addr}')

        openroad_task = OpenRoadTask(testcase_name=task.testcase_name,
                                     task_mode=task.task_mode,
                                     custom_strategies=task.custom_strategies,
                                     parallel_workers=task.parallel_workers,
                                     api_addresses=task.api_addresses,
                                     api_timeout=task.api_timeout,
                                     net_ordering_evaluation_mode=task.net_ordering_evaluation_mode,
                                     droute_end_iter=task.droute_end_iter,
                                     openroad_task_list=openroad_task_list)

        openroad_task.start()

        return resp_success({
            'testcase_name': task.testcase_name,
            'task_mode': task.task_mode,
            'task_full_name': openroad_task.task_full_name,
        })
    except Exception as e:
        print(repr(e))
        return resp_fail(msg=str(e))


@router.delete("/{task_full_name}")
async def terminate_task(task_full_name):
    task = openroad_task_list.get(task_full_name)
    if task is None:
        return resp_fail(msg=f'task {task_full_name} not found')
    task.terminate()
    del openroad_task_list[task_full_name]
    return resp_success()
