import os
import sys
import pandas as pd

from out import *
# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本

# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")


    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 以下区域为预测逻辑代码, 下面的仅为示例
    # 请选手根据实际模型预测情况修改
    testa = pd.read_csv(testa_csv_path) 
    id = testa["id"]
    try:

        result = model_out_cxy(testa_csv_path, testa_html_dir, run_py)

        for i in range(len(id)):
            if result[i] == 1:
                testa.loc[i, "label"] = 1
            else:
                testa.loc[i, "label"] = 0

    except:

        # print("模型预测出错")
        for i in range(len(id)):
            testa.loc[i, "label"] = 0

    test = testa[["id", "label"]]

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
   
    # 结果输出到result_save_path
    test.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
