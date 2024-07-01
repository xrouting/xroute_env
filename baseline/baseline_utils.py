import torch
import math
import zmq
from loguru import logger

from build_3Dgrid import build_3Dgrid
import openroad_api.proto.net_ordering_pb2 as net_ordering

def handle_messange(message,socket):
    '''
    解析后端发送的数据
    '''
    data = None
    if message.HasField('request'):
        req = message.request
        data = [
            [req.dim_x, req.dim_y, req.dim_z],
            [],
            [req.reward_violation, req.reward_wire_length, req.reward_via],
            list(map(lambda x: x + 1, req.nets))  # 在 XRoute 中，net 是从 1 开始的
        ]

        for node in req.nodes:
            node_type = 0
            if node.type == net_ordering.NodeType.ACCESS:
                node_type = node.net + 1  # 在 XRoute 中，net 是从 1 开始的
            elif node.type == net_ordering.NodeType.BLOCKAGE:
                node_type = -1

            node_pin = -1
            #if node.type == net_ordering.NodeType.ACCESS:
            if node.type == net_ordering.NodeType.ACCESS:
                node_pin = node.pin + 1  # 在 XRoute 中，pin 是从 1 开始的

            info = [
                [node.maze_x, node.maze_y, node.maze_z],
                [node.point_x, node.point_y, node.point_z],
                [int(node.is_used), node_type, node_pin],
            ]
            data[1].append(info)        
        if req.is_done:
            socket.send(b'\0')
    return data

def normalize(encoded_state):
    '''对隐藏向量进行规范
    args:
        encoded_state:Tensor.[batch_size,hidden_size]
    return:
        res:Tensor.[batch_size,hidden_size]
    '''
    res = []
    for state in encoded_state:
        min_encoded_state = min(state)
        max_encoded_state = max(state)
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            state - min_encoded_state
        ) / scale_encoded_state
        res.append(encoded_state_normalized.reshape(-1,len(state)))
    res = torch.concat(res)
    return res

def _getNetOrder(order):
    #order:[D,H,W]
    try:
        order = order.numpy()
    except TypeError:
        order = order.cpu()
        order = order.numpy()
    order = order[order>=1].tolist()
    res = []
    for o in order:
        res.append(int(o))
    return res

def getNetOrder(OrderTensor):
    #OrderTensor的尺寸与3D网格对齐尺寸相同,暂时为[N,D=7,H=64,W=64]
    #return:[N,-1]
    netOrder = []
    batch = len(OrderTensor)
    for i in range(batch):
        order = _getNetOrder(OrderTensor[i])
        netOrder.append(order)
    return netOrder    

def padding(matrix,standard=[15,15]):
    length = len(standard)
    assert length in (2,3)
    #限制matrix形如[N,C,H,W] or [N,C,D,H,W]
    assert len(matrix.shape) in (3,4)
    shape = matrix.shape[-length:]
    to_pad = [0] * length
    for idx in range(length):
        if shape[idx] >= standard[idx]:
            pass
        else:
            to_pad[idx] = standard[idx] - shape[idx]
    if to_pad == [0] * length:
        return matrix
    pad = []
    for idx in range(length-1,-1,-1):
        pad.append(0)
        pad.append(to_pad[idx])
    return torch.nn.functional.pad(matrix,pad,"constant",0)  
  

def conv3x3x3(in_channels,out_channels,kernel_size=3,stride=1,padding=1):
    '''3D conv,size remained
    '''
    return torch.nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding)

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    #logger.info(f"&&&sizes:{sizes}")
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def clip(matrix=torch.randn([5,10,30]),standard=[15,15],kernel_size=3,out_channels=None,conv=None):
    '''此函数的目的：通过一次卷积，把输入尺寸削减到小于等于标准尺寸，但是又不能使卷积核的扫描有遗漏
    在Representation中的用法:于forward函数中,设定matrix的channels位置为固定长度,设置conv为对应的channels
    #假定卷积核大小为3,若要使得输出尺寸小于等于标准尺寸：
    #1、当输入尺寸小于等于目标尺寸时,stride最小可以为1
    #2、当输入尺寸大于目标尺寸,但是小于目标尺寸的2倍时,此时stride最小为2
    #3、当输入尺寸大于目标尺寸的2倍,但是小于目标尺寸的3倍时,此时stride最小为3
    #4、当输入尺寸大于目标尺寸的3倍时,stride最小为4
    #   大于卷积核尺寸，此时必定造成卷积核扫描遗漏
    #   依此推论,输入尺寸的尺寸必须大约等于卷积核尺寸（保证卷积可进行）,同时小于等于标准尺寸的kernel_size倍   
    '''
    length = len(standard)
    assert length in (2,3)
    #要求matrix至少为三维，对应[N,H,W],或者四维，对应[N,C,H,W]，或者五维,对应[N,C,D,H,W]
    assert max(3,length) <= len(matrix.shape) <= 5
    if length == 2:
        assert len(matrix.shape) in (3,4)
    else:
        assert len(matrix.shape) == 5
    #if length == 3:
    #    matrix = t.transpose(matrix,2,4)#把[N,C,w,H,D]转置为[N,C,D,H,W]
    shape = matrix.shape[-length:]
    #matrix任一分量不应小于kernel_size
    assert min(shape) >= kernel_size
    #matrix任一分量应小于等于标准尺寸的kernel_size倍
    #assert max(shape) <= kernel_size * min(standard)
    
    #记录输入尺寸超出标准的量
    exceed_length = [0] * length
    for idx in range(length):
        if shape[idx] > standard[idx]:
            exceed_length[idx] = shape[idx]- standard[idx]

    #计算卷积过程各维度的stride
    stride = []
    for idx in range(length):
        stride.append(math.ceil(exceed_length[idx] / standard[idx]) + 1)

    #设置卷积核
    in_channels = None

    matrix_shape = matrix.shape
    #logger.info(f'matrix.shape:{matrix.shape}')
    if len(matrix_shape) == 3:
        in_channels = 1
        matrix = matrix.unsqueeze(1)
        #logger.info(f'matrix.shape:{matrix.shape}')
        if conv is None:
            conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels if out_channels else in_channels,kernel_size=kernel_size,stride=stride)
        else:
            conv.stride = stride
    else:
        in_channels = matrix.shape[1] 
        if conv is None:        
            conv = torch.nn.Conv3d(in_channels=in_channels,out_channels=out_channels if out_channels else in_channels,kernel_size=kernel_size,stride=stride)
        else:
            conv.stride = stride
    try:
        conv_res = conv(matrix)
    except Exception as e:
        import time
        time.sleep(3)
        print('wait 10 seconds')
        print('try again')
        conv_res = conv(matrix)

    array = []
    if len(matrix_shape) in (3,4):
        for i in range(conv_res.shape[0]):
            array.append(padding(matrix=conv_res[i,:,:,:],standard=standard))
        res = torch.concat(array,axis=0).reshape([matrix_shape[0],out_channels if out_channels else in_channels,standard[0],standard[1]])
        return res,stride,res.shape
    
    for i in range(conv_res.shape[0]):
        array.append(padding(matrix=conv_res[i,:,:,:,:],standard=standard))
    res = torch.concat(array,axis=0).reshape([matrix_shape[0],out_channels if out_channels else in_channels,standard[0],standard[1],standard[2]])     
    return res,stride,res.shape

# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = conv3x3x3(num_channels, num_channels)
        self.bn1 = torch.nn.BatchNorm3d(num_channels)
        self.conv2 = conv3x3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm3d(num_channels)  
       

    def forward(self, x):
        #from loguru import logger
        #logger.info(f'before forward,x.shape:{x.shape}')
        out = self.conv1(x)
        #logger.info(f'before bn1,x.shape:{x.shape}')
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out 
    
class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        device=None
    ):
        super().__init__()
        self.device = device

        ##1、对net 3D grid做对齐
        #卷积核输入的固定维度（取决于net的特征数量，暂时为7）
        self.net_input_channels = 7
        self.standard_channels = 1
        self.standard_net_shape = [3,64,64] #[D,H,W]

        #第一层不改变尺寸
        self.net_conv1 = ResidualBlock(num_channels=self.net_input_channels)
        #第二层对齐尺寸,但不改变通道数
        self.net_align_conv1 = torch.nn.Conv3d(in_channels=self.net_input_channels,
                                                out_channels=self.net_input_channels,
                                                kernel_size=5,
                                                stride=1,
                                                padding=1)
        #第三层继续抽取对齐后的特征
        self.net_conv2 = ResidualBlock(num_channels=self.net_input_channels)
        #此时尺寸应当为[N=1,net_input_channels=7,standard_D,standard_H,standard_W]，第四层转换尺寸为(N=1,1,1,1,W)
        self.net_align_conv2 = torch.nn.Conv3d(in_channels=self.net_input_channels,
                                                out_channels=self.standard_channels,
                                                kernel_size=(self.standard_net_shape[0],self.standard_net_shape[1],3),
                                                stride=1,
                                                padding=(0,0,1))
        


        ##2、对obstacle 3D grid做对齐
        self.ob_input_channels = 1
        self.standard_obs_shape = []
        #第一层不改变尺寸
        self.ob_conv1 = ResidualBlock(num_channels=self.ob_input_channels)
        #第二层对齐尺寸,但不改变通道数
        self.ob_align_conv1 = torch.nn.Conv3d(in_channels=self.ob_input_channels,
                                                out_channels=self.net_input_channels,
                                                kernel_size=5,
                                                stride=1,
                                                padding=1)
        #第三层继续抽取对齐后的特征
        self.ob_conv2 = ResidualBlock(num_channels=self.net_input_channels)
        #此时尺寸应当为[N,ob_input_channels=1,standard_D,standard_H,standard_W]，第四层转换尺寸为(N,1,1,1,W)
        self.ob_align_conv2 = torch.nn.Conv3d(in_channels=self.net_input_channels,
                                                out_channels=self.ob_input_channels,
                                                kernel_size=(self.standard_net_shape[0],self.standard_net_shape[1],3),
                                                stride=1,
                                                padding=(0,0,1))    

    def forward(self,x):
        '''
        args:
            x:numpy.array.即环境的observation.维度为[N,不定长的channel,D,H,W],第一个channnel是obstaclesGrid,
                第二个表示netOrder,第三个开始,每7个channel代表一个netGrid
        res:
            representation_list:torch.Tensor.维度是[N,hidden_dim]
            action_mapping_list:list[dict[key:int,value:Tensor]].每个value维度是[hidden_dim]
            需要转成[N,Channel,hidden_size],第一个channel编码Net序号,prediction函数再将其解码成字典
        '''
        n_channel_per_net = 7
        clip_kernel_size = 5

        ###处理obstacles
        obs_list = []#[N,W]
        for i in range(len(x)):
            obstaclesGrid = torch.Tensor(x[i][0]).unsqueeze(0).unsqueeze(0)#[1,1,D,H,W]
            obs = self.ob_conv1(obstaclesGrid.to(self.device))
            obs,stride,_  = clip(obs,
                                    standard=self.standard_net_shape,
                                    kernel_size=clip_kernel_size,
                                    out_channels=self.net_input_channels,
                                    conv=self.ob_align_conv1)    
            try:
                assert max(stride) <= clip_kernel_size,"stride:%s" % stride
            except Exception as e:
                print(f'obs.shape:{obs.shape}')
                print(e)
            obs = self.ob_conv2(obs)
            #转换尺寸为(N,1,1,1,W)
            obs = self.ob_align_conv2(obs)
            #转换尺寸为[hdden_dim = W]
            obs = obs.reshape([obs.shape[-1]])
            obs_list.append(obs)
        

        ###分别处理每个net集合
        orderList = []
        for i in range(len(x)):
            orderList.append(torch.Tensor(x[i][1]).unsqueeze(0))#N个[1,D,H,W]
        netOrder = getNetOrder(orderList)#[N,-1]

        netGridTensor = []
        for i in range(len(x)):
            netGridTensor.append(torch.Tensor(x[i][2:]))#[N,C,D,H,W]       
        for i in range(len(netGridTensor)):
            assert netGridTensor[i].shape[0] // n_channel_per_net == len(netOrder[i]),"netGridTensor[i].shape[0] // %s = %s,len(netOrder[i]) = %s" %(n_channel_per_net,netGridTensor[i].shape[0] // n_channel_per_net,len(netOrder[i]))
        
        action_mapping_list = []#N维列表,每个元素是一个字典
        #batch loop
        for b in range(len(netGridTensor)):
            order = netOrder[b]
            gridTensor = netGridTensor[b].unsqueeze(0)#[N=1,不定长C,D,H,W]
            nNet = gridTensor.shape[1] // n_channel_per_net
            assert len(order) == nNet,'len(order) = %s,nNet = %s' %(len(order,nNet))

            #存储当前episode所有net的特征
            episode_net_list = []
            #存储action字典,每个action对应一个net，一个net对应一个定长向量
            action_rep_mapping = {}            
            #net loop:把每个net逐个encode到三维
            for i in range(nNet):
                net = gridTensor[:,i*n_channel_per_net:(i+1)*n_channel_per_net,:,:,:].to(self.device)#[N=1,C=n_channel_per_net,D,H,W]
                net = net[:,:n_channel_per_net,:,:,:]#[N=1,C=n_channel_per_net,D,H,W]

                #第一层不改变尺寸
                net = self.net_conv1(net)#[N=1,C=n_channel_per_net - 1,D,H,W]

                #第二层对齐尺寸,但不改变通道数
                net,stride,_ = clip(net,
                                    standard=self.standard_net_shape,
                                    kernel_size=clip_kernel_size,
                                    out_channels=self.net_input_channels,
                                    conv=self.net_align_conv1)#[N=1,C=n_channel_per_net - 1,D_standard,H_standard,W_standard]
                try:
                    assert max(stride) <= clip_kernel_size,"stride:%s" % stride
                except Exception as e:
                    print(f'net.shape:{net.shape}')
                    print(e)                

                #第三层继续抽取对齐后的特征
                net = self.net_conv2(net)#[N=1,C=n_channel_per_net - 1,D_standard,H_standard,W_standard]

                #第四层改变形状,并转换为定长向量
                net = self.net_align_conv2(net)#(N=1,1,1,1,W_standard)
                net_shape = net.shape#[N=1,1,1,1,W]
                net = net.reshape([net_shape[-1]])#[W_standard]

                #记录每个net对应的定长向量,即每个net的特征向量
                action_rep_mapping[order[i]] = net#[W_standard]
                episode_net_list.append(net)

            action_mapping_list.append(action_rep_mapping)

        representation_list = obs_list
        return representation_list,action_mapping_list    



class Game:
    """
    Game wrapper.
    """
    def __init__(self,port_recv='5556',port_initial='6667'):
        self.socket = None
        self.port_recv = port_recv
        self.port_initial = port_initial

    def step(self,action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """        
        print(f'action={action}')
        done = False
        observation = None
        #发送action
        context = zmq.Context()
        if not self.socket:
            self.socket = context.socket(zmq.REP)
            self.socket.bind('tcp://*:'+ self.port_recv)          
        message = net_ordering.Message()      
        message.response.net_index = int(action) - 1   
        self.socket.send(message.SerializeToString()) 
        self.routed_nets.add(action)
        print(f'send action:{int(action)}')

        #接收新的环境信息并解析
        message_raw = self.socket.recv()   
        message = net_ordering.Message()
        message.ParseFromString(message_raw)
        data = handle_messange(message,self.socket)

        #把环境信息构建为3D特征网格
        observation,netSet,\
            self.violation_cur_step,self.wirelength_cur_step,self.via_cur_step = build_3Dgrid(data,self.routed_nets)
        
        #计算当前步骤的指标
        violation = self.violation_cur_step - self.violation_last_step
        via = self.via_cur_step - self.via_last_step
        wirelength = self.wirelength_cur_step - self.total_wirelength_last_step

        #更新上一步的指标记录
        self.total_wirelength_last_step = self.wirelength_cur_step
        self.via_last_step = self.via_cur_step
        self.violation_last_step = self.violation_cur_step

        if len(netSet) == 0:
            done = True

        self.legal_action_set = netSet
        return observation,done,violation,wirelength,via

    def reset(self):
        """Reset the game for a new game.
        Returns:
            observation:numpy.array.[N=1,C,D,H,W].Initial observation of the game.
            reset_try_time:int.
        """
        ###向后台发送初始化请求
        done = True
        reset_try_time = 0
        while done:            
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect('tcp://127.0.0.1:'+self.port_initial)  

            #通知后台初始化GCell
            socket.send(b'initial') 

            #接收后台的初始环境信息
            context = zmq.Context()
            if not self.socket:
                self.socket = context.socket(zmq.REP)
                self.socket.bind('tcp://*:'+self.port_recv)
            message_raw = self.socket.recv()

            #解析消息
            message = net_ordering.Message()
            message.ParseFromString(message_raw)
            data = handle_messange(message,self.socket)

            #把初始环境构建成3D特征网格
            self.routed_nets = set()
            self.observation,self.action_space,\
                self.violation_last_step,self.total_wirelength_last_step,self.via_last_step = build_3Dgrid(data,self.routed_nets)

            if len(self.action_space) != 0:
                done = False
            else:
                #如果当前是没有待布网络的GCell，则继续接收下一个GCell
                reset_try_time += 1
        print(f'self.observation.shape:{self.observation.shape}')
        return self.observation,reset_try_time

