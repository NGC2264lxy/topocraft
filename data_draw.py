import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import math
from matplotlib.ticker import PercentFormatter

fct_file_pattern = 'D:/meteor/topoCrafter_code/data/datadraw/FCT/FCT*.txt'  
p99fct_file_pattern = 'D:/meteor/topoCrafter_code/data/datadraw/P99FCT/P99FCT*.txt'  
general_file_pattern = 'D:/meteor/topoCrafter_code/data/datadraw/General/General-#*.sca'  

traffic_Send_file_pattern = 'D:/meteor/topoCrafter_code/data/datadraw/Traffic Demand/Traffic Demand*.txt'  


fct_files = glob.glob(fct_file_pattern)
p99fct_files = glob.glob(p99fct_file_pattern)
general_files = glob.glob(general_file_pattern)

traffic_files=glob.glob(traffic_Send_file_pattern)

offer_loads = [0.1 * (i + 1) for i in range(len(fct_files))]  # 根据文件数量生成 offer load 值
average_fcts = []
average_p99fcts = []
average_throughoutputs=[]
average_flitEndToEndDelays=[]
average_link_utilizations=[]
portsends=[]
max_link_utilizations=[]

# FCT、P99FCT
for fct_file in fct_files:
    # 读取文件，假设每行是一个 FCT 值
    fct_values = pd.read_csv(fct_file, header=None, names=['FCT'])
    
    # 计算平均 FCT 值
    average_fct = (fct_values['FCT'].mean())
    print(average_fct)
    average_fcts.append(average_fct*1e6)

for p99fct_file in p99fct_files:
    # 读取文件，假设每行是一个 P99FCT 值
    p99fct_values = pd.read_csv(p99fct_file, header=None, names=['P99FCT'])
    
    # 计算平均 P99FCT 值
    average_p99fct = p99fct_values['P99FCT'].mean()
    
    average_p99fcts.append(average_p99fct*1e6)
    print(f"average_p99fcts:{average_p99fcts}")
    
for general_file in general_files:
    flitEndToEndDelays=[]
    with open(general_file, 'r') as f:
        for line in f:
            # if 'flitEndToEndDelay:mean' in line:
            #     # 使用正则表达式提取数字部分
            #     match1 = re.search(r'[-+]?\d*\.\d+|\d+', line)
            #     if match1:
            #         flitEndToEndDelay= float(match1.group())
                    
            #         flitEndToEndDelays.append(flitEndToEndDelay)
            #         #所有ALL_MSG_RECEIVE在一个文件中是一样的，取一个即可
            if 'flitEndToEndDelay:mean' in line:
                parts = line.split()  # 根据空格分割
                flitEndToEndDelay = float(parts[-1])  # 假设值是最后一个部分
                if math.isnan(flitEndToEndDelay):
                    flitEndToEndDelay = 0
                flitEndToEndDelays.append(flitEndToEndDelay)
              

            if 'ALL_MSG_RECEIVE' in line:
                parts = line.split()  # 根据空格分割
                all_msg_receive = int(parts[-1])  # 假设值是最后一个部分
                # print(f'ALL_MSG_RECEIVE: {all_msg_receive}')
                               
                 
# 时延
    if flitEndToEndDelays:
        print(f'FLIT_ENDTOEND_DELAY: {flitEndToEndDelays}')
        average_flitEndToEndDelays.append((sum(flitEndToEndDelays) / len(flitEndToEndDelays))*1e6)
        print(f"length:{len(flitEndToEndDelays)}")
        print(f'AVERAGE_FLIT_ENDTOEND_DELAY: {average_flitEndToEndDelays}')
#吞吐        
    
for general_file in general_files:
    with open(general_file, 'r') as f:
        for line in f:
            if 'ALL_MSG_RECEIVE' in line:
                parts = line.split()  # 根据空格分割
                all_msg_receive = int(parts[-1])  # 假设值是最后一个部分
                break
        if all_msg_receive:
            print(f"all:{all_msg_receive}")
            average_throughoutputs.append((all_msg_receive*256*64*8)/(0.01*1e9))  
            print(f"average_throughoutputs:{average_throughoutputs}")      

# 最大链路利用率、平均链路利用率
# for traffic_file in traffic_files:
#    #trafficsend矩阵的数
#    TrafficSend= pd.read_csv(traffic_file, header=None, names=['TrafficSend'])
#    max_TrafficSend=TrafficSend['TrafficSend'].max()
#    print(f"max_TrafficSend：{max_TrafficSend}")
#    max_link_utilization=(max_TrafficSend*64*8)/(100*1e9*0.01*2) #0.01是10ms  port5,6两个口
#    print(f"max_link_utilization：{max_link_utilization}")
#    max_link_utilizations.append(max_link_utilization)
#    average_Trafficsend=TrafficSend['TrafficSend'].mean()
#    average_link_utilization=(average_Trafficsend*64*8)/(100*1e9*0.01*2) #0.01是10ms port5,6两个口
#    average_link_utilizations.append(average_link_utilization)

# 最大链路利用率、平均链路利用率
for general_file in general_files:
    with open(general_file, 'r') as f:
        for line in f:
            if 'PortSend_5' in line:
                parts = line.split()  # 根据空格分割
                TrafficSend_port5 = int(parts[-1])  # 假设值是最后一个部分
                portsends.append(TrafficSend_port5)
            if 'PortSend_6' in line:
                parts = line.split()  # 根据空格分割
                TrafficSend_port6 = int(parts[-1])  # 假设值是最后一个部分
                portsends.append(TrafficSend_port6)
        max_link_utilization=(max(portsends)*64*8)/(100*1e9*0.01) #0.01是10ms  port5,6两个口
        max_link_utilizations.append(max_link_utilization)
        average_value = sum(portsends) / len(portsends) if portsends else 0
        average_link_utilization=(average_value*64*8)/(100*1e9*0.01) #0.01是10ms port5,6两个口
        average_link_utilizations.append(average_link_utilization)


fig, axs = plt.subplots(2,3, figsize=(14, 6))

# 绘制平均 FCT
axs[0,0].plot(offer_loads, average_fcts, marker='o',color='green')
axs[0,0].set_title('Average FCT (us)')
axs[0,0].set_xlabel('Offer Load')
axs[0,0].set_ylabel('Average FCT')
axs[0,0].grid()
axs[0,0].set_xticks(offer_loads)

# 绘制 Throughouput
axs[0,1].plot(offer_loads, average_throughoutputs, marker='o', color='red')
axs[0,1].set_title('Throughouput(Gbps)')
axs[0,1].set_xlabel('Offer Load')
axs[0,1].set_ylabel('average_throughoutputs')
axs[0,1].grid()
axs[0,1].set_xticks(offer_loads)

# 绘制 P99 FCT
axs[0,2].plot(offer_loads, average_p99fcts, marker='o', color='orange')
axs[0,2].set_title('P99 FCT (us)')
axs[0,2].set_xlabel('Offer Load')
axs[0,2].set_ylabel('P99 FCT')
axs[0,2].grid()
axs[0,2].set_xticks(offer_loads)



# 绘制 Delay

axs[1,0].plot(offer_loads,average_flitEndToEndDelays, marker='o', color='red')
axs[1,0].set_title('Delay(us)')
axs[1,0].set_xlabel('Offer Load')
axs[1,0].set_ylabel('Delay')
axs[1,0].grid()
axs[1,0].set_xticks(offer_loads)

# 绘制 Max_Link_Utilization
# print(f"max_link_utilizations:{max_link_utilizations}")
axs[1,2].plot(offer_loads, max_link_utilizations, marker='o', color='orange')
axs[1,2].set_title('Max Link Utilization ')
axs[1,2].set_xlabel('Offer Load')
axs[1,2].set_ylabel('Max Link Utilization')
axs[1,2].grid()
axs[1,2].set_xticks(offer_loads)
axs[1,2].yaxis.set_major_formatter(PercentFormatter(xmax=1))  # x_max 是数据的最大值（在 0 到 1 的范围内）

# 绘制 Average_Link_Utilization
axs[1,1].plot(offer_loads, average_link_utilizations, marker='o', color='orange')
axs[1,1].set_title('Average Link Utilization ')
axs[1,1].set_xlabel('Offer Load')
axs[1,1].set_ylabel('Average Link Utilization')
axs[1,1].grid()
axs[1,1].set_xticks(offer_loads)
axs[1,1].yaxis.set_major_formatter(PercentFormatter(xmax=1))  # x_max 是数据的最大值（在 0 到 1 的范围内）

# 调整布局
plt.tight_layout()
plt.show()

