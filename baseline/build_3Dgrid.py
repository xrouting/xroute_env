import torch as t

def get_grid_size(data):
    return data[0]

def getObstaclesAndAccessPoints(data,routed_nets,bool_inference):
    '''获得所有的障碍点和Access point(一个pin可能包含多个Access point)
    args:
        data:list.包含布局所有网格节点的信息
        routed_nets:set.已经布了的网络集合
        bool_inference:bool.若为True,则为推断模式,否则为训练模式
    return:
        obstacles:list.每个元素是三元组坐标
        accessPoints:dict.键为待布网络网络编号,值为list.形如[(x_1,y_1,z_1),...,(x_n,y_n,z_n)]
    '''
    obstacles = []
    accessPoints = {}
    for vertex in data[1]:
        bool_occupy,Net,Pin = vertex[2]
        if Net == -1:
            obstacles.append(vertex)
        elif Net == 0:
            #普通点被占用，视同obstacle
            if bool_occupy == 1:
                obstacles.append(vertex)
        else:
            '''
            #注意,假设一个网络已经被布了,但是由于一个pin存在多个access point,此时这个网络也会被统计在accessPoints中
            #这种方式仅在布局初始化时才能准确得到所有待布线网络
            '''
            #vertex是pin点的情况
            assert type(Net) is int and Net >= 1
            #pin点被占用，视同obstacle
            if bool_occupy == 1:
                obstacles.append(vertex)            
            if Net in accessPoints:
                if Pin not in accessPoints[Net]:
                    accessPoints[Net][Pin] = [vertex]
                else:
                    accessPoints[Net][Pin].append(vertex)
            else:
                accessPoints[Net] = {}
                accessPoints[Net][Pin] = [vertex]

    #训练模式下,删除掉已经布的网络
    if not bool_inference:
        for netId in routed_nets:
            '''
            #之所以要if netId in accessPoints的原因:
            #假设A网络要布线,A网络每一个pin点只有一个access point.
            #布线A以后,会导致A的所有access point全部为占用状态,那么accessPoints将不会包含A网络这个键,
            #如果直接执行删除就会报错.
            '''
            if netId in accessPoints:
                del accessPoints[netId]
    return obstacles,accessPoints


def _get_adjacent_point(vertex,direction,grid_length,grid_width,grid_height):
    assert direction in ('east','west','south','north','up','down')
    if direction == 'west':
        if vertex[0] == 0:
            return -1
        else:
            return (vertex[0] - 1,vertex[1],vertex[2])
    if direction == 'east':
        if vertex[0] == grid_length - 1:
            return -1
        else:
            return (vertex[0] + 1,vertex[1],vertex[2] )

    if direction == 'south':
        if vertex[1] == 0:
            return -1
        else:
            return (vertex[0],vertex[1]-1,vertex[2])
    if direction == 'north':
        if vertex[1] == grid_width - 1:
            return -1
        else:
            return (vertex[0],vertex[1]+1,vertex[2] )

    if direction == 'down':
        if vertex[2] == 0:
            return -1
        else:
            return (vertex[0],vertex[1],vertex[2]-1)
    if direction == 'up':
        if vertex[2] == grid_height - 1:
            return -1
        else:
            return (vertex[0],vertex[1],vertex[2]+1)

def getObstacleGrid(obstacles,grid_dim):
    grid = []
    #obstacles 0-1 features
    array = t.zeros(grid_dim)
    for ob in obstacles:
        idx = ob[0]
        array[idx[0]][idx[1]][idx[2]] = 1
    grid.append(array)
    #[N,Channel,Depth,Height,Width]
    return t.concat(grid,axis=0).reshape(-1,1,grid_dim[2],grid_dim[1],grid_dim[0])


def getNetGrid(pinDict,grid_dim):
    '''
    args:
        pinDict:dict[pinId:[accessPointList]].同一个net的所有pin的所有access points
        grid_dim:list or tuple.指定3D网格的尺寸,分别对应[W,H,D]
    '''
    #access points 0-1 features
    netAccessPoints = set()
    grid = []
    array = t.zeros(grid_dim)
    for accesspointList in pinDict.values():
        for accesspoint in accesspointList:
            idx = accesspoint[0]
            array[idx[0]][idx[1]][idx[2]] = 1
            netAccessPoints.add( (idx[0],idx[1],idx[2]) )
    grid.append(array)

    #whether adjacent point belong to a same pin(0-1 feature)
    #directions' oder: ('east','south','west','north','up','down')
    array = [t.zeros(grid_dim)] * 6
    for ap in netAccessPoints:
        for idx,direction in enumerate(['east','south','west','north','up','down']):
            #获取当前access point的相邻点坐标，若返回-1则说明不存在
            adjacent_point = _get_adjacent_point(vertex=ap,
                                                    direction=direction,
                                                    grid_length=grid_dim[0],
                                                    grid_width=grid_dim[1],
                                                    grid_height=grid_dim[2])

            if adjacent_point != -1 and adjacent_point in netAccessPoints:
                    #若相邻点也是access point
                    #array[idx][adjacent_point[0]][adjacent_point[1]][adjacent_point[2]] = 1
                    array[idx][ap[0]][ap[1]][ap[2]] = 1

    grid.extend(array)
    #[N,C,D,H,W]
    return t.concat(grid,axis=0).reshape(len(grid),grid_dim[2],grid_dim[1],grid_dim[0])

def getNetOrderChannel(obstaclesGrid,netOrder):
    netOrderChannel = t.zeros_like(obstaclesGrid)#[1,1,D,H,W]
    shape = netOrderChannel.shape
    H,W = shape[3],shape[4]
    d,h,w= 0,0,0
    net_counter = 0
    for netId in netOrder:
        netOrderChannel[0][0][d][h][w] = netId
        if (net_counter + 1) % (H*W) == 0:
            d += 1
            h,w = 0,0
        elif (net_counter + 1) % W == 0:
            h += 1
            w = 0
        else:
            w += 1
        net_counter += 1
    return netOrderChannel    

def _build_3Dgrid(obstaclesGrid,netGridDict):
    '''
    args:
        obstaclesGrid:Tensor.[N=1,C,D,H,W]
        netGridDict:set.{NetId:Tensor}.Tensor的维度为[C,D,H,W]
    return:
        observation:Tensor.[N=1,C=2+len(netGridDict)*8,D,H,W](包含obstaclesGrid,netOrderChannel,netGrid)
            或[N=1,C=2,D,H,W](仅包含obstaclesGrid,netOrderChannel)
        netSet:set.
    '''
    netSet = set(netGridDict.keys())
    netOrder = []
    netArray = []
    #初始化时,网络次序为按编号升序
    for netId in sorted(list(netSet)):
        netOrder.append(netId)
        netArray.append(netGridDict[netId].unsqueeze(0))

    #创建一个channel来记录待布线网络，用于关联observation的哪几个通道对应该网络的特征
    netOrderChannel = getNetOrderChannel(obstaclesGrid,netOrder)          
    netGrid = t.concat(netArray,axis=1) if len(netArray) > 0 else []
    if netGrid != []:
        observation = t.cat([obstaclesGrid,netOrderChannel,netGrid],dim=1)
    else:
        observation = t.cat([obstaclesGrid,netOrderChannel],dim=1)
    return observation,netSet

def getNetTensor(netIndex,observation):
    netsTensor = observation[:,2:,:,:,:]
    return netsTensor[:,netIndex*8:(netIndex+1)*8,:,:,:]

def orderNets(observation,netOrder):
    '''根据netOrder对observation的channel调整顺序
    args:
        observation:numpy.array.[N,不定长的channel,D,H,W]
        netOrder:dict.(key:action,value:probabilty of the action)
    return:
        sorted_observation:
    '''
    #按概率大小对action进行升序
    sorted_nets = sorted(netOrder.items(),key=lambda x:x[1])
    sorted_actions = [i[0] for i in sorted_nets]

    #observation中netId对应的顺序
    netIndexDict = {}
    actions = sorted(list(netOrder.keys()))
    for idx,action in enumerate(actions):
        netIndexDict[action] = idx

    obstaclesGrid = observation[:1,:1,:,:,:]#[N=1,C=1,D,H,W]
    print(f'sorted_actions:{sorted_actions}')
    netOrderChannel = getNetOrderChannel(obstaclesGrid,sorted_actions) 
    netArray = []
    for netId in sorted_actions:
        netIndex = netIndexDict[netId]
        curNetTensor = getNetTensor(netIndex,observation)
        netArray.append(curNetTensor)
    netGrid = t.concat(netArray,axis=1) 
    sorted_observation = t.cat([obstaclesGrid,netOrderChannel,netGrid],dim=1)
    return sorted_observation       
    
def build_3Dgrid(data,routed_nets,bool_inference=False):
    '''
    args:
        data:list.OpenRoute传过来的数据
        routed_nets:set.已经布了的网络集合,需要这个变量的原因是,仅在布局初始化时,从数据本身统计得到的待布线网络才是完整的。
            在训练过程中,假设有ABC三个网络需要布线,有可能布完A和B以后,C的所有pin点也已经被占用了,但是C并没有连通。
            这种时候如果只根据pin点的占用信息来判断C是否已经布线,会得到相反结果。
            因此训练模式下需要在每次布局开始时用一个集合来跟踪记录实际已经布了的网络。
        bool_inference:bool.如果为true,若为True,则为推断模式,.否则为训练模式。
            注意,推断模式下data中会包含待布线的网络列表。
            训练模式下需要从data每一个数据点整理统计出需要布哪些网络。
    res:
        observation:numpy.array.[N=1,C,D,H,W].此处标明N=1是因为,无论训练还是推断,本函数仅用于一个布局数据的解析
        netSet:set.
        reward:float
    '''
    #先获得所有的障碍点和Access point(一个pin可能包含多个Access point)
    obstacles,accessPoints = getObstaclesAndAccessPoints(data,routed_nets,bool_inference) 

    if bool_inference:
        #获得推断模式下发送过来的待布网络
        netList = data[3]
        print(f'receive netList={netList}')
        keys = list(accessPoints.keys())
        for key in keys:
            if key not in netList:
                del accessPoints[key]
                
    #获取GCell的维度
    grid_dim = get_grid_size(data) 

    #把障碍点信息整理成一个3D特征网格
    obstaclesGrid = getObstacleGrid(obstacles,grid_dim)

    #把每个net(即每个action)的数据组织成有7个通道的3D特征网格
    netGridDict = {}
    print(f'build_3Dgrid,netId:{sorted(list(accessPoints.keys()))}')
    for netId,pinDict in accessPoints.items():
        netGridDict[netId] = getNetGrid(pinDict,grid_dim)

    #把obstaclesGrid和netGridDict整合在一起形成完整的3D特征网格,
    # 并且设置一个特殊的通道用以记录网络的序号
    observation,netSet =_build_3Dgrid(obstaclesGrid,netGridDict)
    #data[2]中包含有violation wirelength,via三项信息
    #但要注意传送回来的指标信息是累加式的,比如从初始状态开始,先布A网络,有2个violation,
    #接着布B网络,返回5个violation,实际上B网络导致的violation是(5-3)个.
    return observation,netSet,data[2][0],data[2][1],data[2][2]

def _getNetOrder(order):
    #order:[D,H,W]
    order = order.numpy()
    order = order[order>=1].tolist()
    res = []
    for o in order:
        res.append(int(o))
    return res

def getNetOrder(OrderTensor):
    #OrderTensor的尺寸与3D网格对齐尺寸相同
    #return:[N,-1]
    netOrder = []
    batch = len(OrderTensor)
    for i in range(batch):
        order = _getNetOrder(OrderTensor[i])
        netOrder.append(order)
    return netOrder


    

