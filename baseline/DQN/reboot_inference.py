import os,time
PORTS = list(range(6654,6666))

def read_last_line(path):
    with open(path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 0:
            return lines[-1]

while True:
    for port in PORTS:
        path = './log'+str(port)+'.txt'
        line = read_last_line(path)
        if line and 'torch.cuda.OutOfMemoryError: CUDA out of memory.' in line:
            print('start rebooting inference')
            os.system('bash inference.sh')
            print('done')
            time.sleep(10)
            break
    time.sleep(10)
            

