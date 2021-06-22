import numpy as np

def compute_meanandvar():
    valuelist=[]
    while True:
        try:
            value = float(input("请输入列表中的数字，其他字符结束："))
            valuelist.append(value)

        except:
            meanvalue=np.mean(valuelist)
            stdvalue=np.std(valuelist)
            return meanvalue,stdvalue

if __name__ == "__main__":
    mean,std=compute_meanandvar()
    print(mean,std)