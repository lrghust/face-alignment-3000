import random
l=open('300w_cropped/full_img_list').readlines()
random.shuffle(l)
with open('300w_cropped/train_img_list', 'w') as f:
    f.writelines(l[:450])
with open('300w_cropped/test_img_list', 'w') as f:
    f.writelines(l[450:])