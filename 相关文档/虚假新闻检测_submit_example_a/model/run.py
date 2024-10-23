import os
import sys
import cv2
import jieba
import requests
import numpy as np
import pandas as pd
# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本

# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")
    testa_image_dir = os.path.join(to_pred_dir, "testa_x", "image")

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 以下区域为预测逻辑代码, 下面的仅为示例
    # 请选手根据实际模型预测情况修改
    
    testa = pd.read_csv(testa_csv_path)
    id = testa["id"]
    official_account_name = testa["Ofiicial Account Name"]
    title = testa["Title"]
    news_url = testa["News Url"]
    image_url = testa["Image Url"]
    report_content = testa["Report Content"]
    testa["label"] = 0

    i = 0
    id = testa.loc[i, "id"]
    # new_url_i = testa.loc[i, "News Url"]
    html_i = os.path.join(testa_html_dir, "%s.html"%id)
    # image_url_i = testa.loc[i, "Image Url"]
    image_i = os.path.join(testa_image_dir, "%s.png"%id)

    html = open(html_i, "r").read().strip()
    print(html)

    image = cv2.imread(image_i)
    print(image.shape)

    testa.loc[0, "label"] = 1

    test = testa[["id", "label"]]

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
   
    # 结果输出到result_save_path
    test.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
