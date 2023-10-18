from fastapi.responses import ORJSONResponse


def resp_success(data=None, msg=None, code=0):
    return ORJSONResponse([{
        'data': data,
        'msg': msg,
        'code': code,
    }])


def resp_fail(msg=None, code=-1, data=None):
    return ORJSONResponse([{
        'msg': msg,
        'code': code,
        'data': data,
    }])
