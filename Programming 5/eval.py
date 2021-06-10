import json

pred_result=json.load(open("val1.json",encoding='utf-8'))
gt_result=json.load(open("./data/val.json",encoding='utf-8'))

TP = 0
all = 0
for key in gt_result:
    n = len(gt_result[key])
    all += n
    for i in range(n):
        if key in pred_result and len(pred_result[key]) > i:
            if gt_result[key][i] == pred_result[key][i]:
                TP += 1

print("Accuracy", round(TP / all, 3))