import pandas as pd
def load_data(fr_file,fw_name):
    data_frame = pd.read_table(fr_file,names=["user_id","item_id","timestamp"],engine="python")
    df = data_frame.sort_values(by="timestamp",ascending=False)
    for line in df.itertuples():
        user = line[1]
        item = line[2]
        time = line[3]
        fw_name.write(str(user)+"\t"+str(item)+"\t"+str(time)+"\n")
        # print(user,item,time)
    # print(data_frame)
    # print(df)


if __name__ == '__main__':
    filepath = "data/ml/training_sort.txt"
    filename = open(filepath,'w')
    load_data("data/ml/training.txt",filename)
    filename.close()