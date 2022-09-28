import pandas as pd
import argparse
# 模型结果融合
def merge(client_id,model_num):
    # print(client_id,model_num)
    res_list = []
    pre_name = 'save/prediction/submit_'+str(client_id)+'_'
    for i in range(model_num):
        file_name = pre_name+str(i)+'.csv'
        res_list.append(file_name)
    # print(res_list)
    res = None
    for file in res_list:
        data = pd.read_csv(file,header=None)
        if res is None :
            res = data[2].tolist()
        else :
            for i,x in enumerate(data[2].tolist()):
                res[i] += x
    data = pd.read_csv(res_list[0],header=None)
    final = []
    for i in range(len(res)):
        if res[i] > model_num/2:
            final.append(1)
        else:
            final.append(0)
    data[2] = final 
    data.to_csv('save/prediction/submit_'+str(client_id)+'.csv',index=False,header=False)
    # print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int)
    parser.add_argument('--model_num', type=int)
    args = parser.parse_args().__dict__
    merge(args['client_id'],args['model_num'])