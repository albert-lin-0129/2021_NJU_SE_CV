import os
import cv2.cv2
from tqdm import tqdm


def tracker(img_dir: str, gt_dir: str):
    cv2_tracker = cv2.TrackerCSRT_create()
    gt = []
    imgs = os.listdir(img_dir)
    gt_txt = os.path.join(img_dir, 'groundtruth.txt')
    with open(gt_txt, 'r') as f:
        for line in f.readlines():
            line = list(map(float, line.strip().split(',')))
            gt.append(list(map(int, line)))
    imgs.sort()
    frame = cv2.imread(os.path.join(img_dir, imgs[0]))
    bbox = gt[0]
    bbox = [bbox[0], bbox[1], bbox[4] - bbox[0], bbox[5] - bbox[1]]
    print(bbox)
    cv2_tracker.init(frame, bbox)
    pr = [gt[0]]
    not_hit_count = fps = 0
    for img in tqdm(imgs):
        if img.endswith('.jpg'):
            frame = cv2.imread(os.path.join(img_dir, img))
            timer = cv2.getTickCount()
            ok, bbox = cv2_tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ok:
                x1 = bbox[0]
                y1 = bbox[1]
                w = bbox[2]
                h = bbox[3]
                pr.append([x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h])
            else:
                not_hit_count += 1
                pr.append([0, 0, 0, 0, 0, 0, 0, 0])

    print('fps: ' + str(fps))
    print('not hit count ' + str(not_hit_count))
    with open(gt_dir, 'w+') as f:
        for p in pr:
            f.write(','.join(str(x) for x in p))
            f.write('\n')


if __name__ == '__main__':
    img_base = 'data/test_public'
    pr_base = 'data/test_predict'
    img_dirs = os.listdir(img_base)
    print('------------------------start------------------------')
    if not os.path.exists(os.path.join(pr_base, '181250084')):
        os.makedirs(os.path.join(pr_base, '181250084'))
    for img_dir in img_dirs:
        print('start to resolve dir: ', img_dir)
        pr_dir = os.path.join(pr_base, '181250084') + '/' + img_dir + '.txt'
        try:
            tracker(os.path.join(img_base, img_dir), pr_dir)
        except:
            continue
    print('------------------------start------------------------')
