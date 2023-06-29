from fastapi import APIRouter

from ..utils.fastapi import resp_success, resp_fail
from ..utils.openroad import OpenRoadTask

router = APIRouter()

task_list = {}


@router.get("/")
async def get_tasks():
    return resp_success(data=list(task_list.keys()))


@router.post("/")
async def create_task():
    try:
        task = OpenRoadTask(testcase_name='ispd18_test1')
        task.start()
        task_list[task.task_full_name] = task
        return resp_success({
            'task_full_name': task.task_full_name,
        })
    except Exception as e:
        return resp_fail(msg=str(e))


@router.delete("/{task_full_name}")
async def terminate_task(task_full_name):
    task = task_list.get(task_full_name)
    if task is None:
        return resp_fail(msg=f'task {task_full_name} not found')
    task.terminate()
    del task_list[task_full_name]
    return resp_success()
