import cv2
import numpy as np
from skimage import transform
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from scipy.io import loadmat
import pickle
import gc
import time

################### params ###################
param_augment_num=5
param_local_feature_num=[500, 500, 500, 300, 300, 200, 200, 200, 100, 100]
param_local_radius=[0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05]
param_landmark_num=68
param_tree_num=10
param_tree_depth=5
param_cascade_num=7

################### shape ###################
# load shape points file
def ReadShape(path):
    file=open(path).readlines()
    shape=[]
    for i in range(68):
        pair=file[i+3].split()
        shape.append([float(pair[0]), float(pair[1])])
    return np.array(shape)

# transform point to [-1,1] relative to center
def Shape2Relative(shape, bbox):
    w=bbox[2]-bbox[0]
    h=bbox[3]-bbox[1]
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    rshape=shape.copy()
    rshape[:, 0]=(rshape[:, 0]-cx)*2/w
    rshape[:, 1]=(rshape[:, 1]-cy)*2/h
    return rshape

# transform to absolute coord
def Shape2Absolute(rshape, bbox):
    w=bbox[2]-bbox[0]
    h=bbox[3]-bbox[1]
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    ashape=rshape.copy()
    ashape[:, 0]=ashape[:, 0]*w/2+cx
    ashape[:, 1]=ashape[:, 1]*h/2+cy
    return ashape

# make the mean shape as (0,0) 
def CenterShape(shape):
    cshape=shape.copy()
    cshape-=cshape.mean(0)
    return cshape

######################## bbox ##########################
def LoadBBox():
    files=['bounding_boxes_afw.mat', 'bounding_boxes_lfpw_trainset.mat', 'bounding_boxes_helen_trainset.mat']
    dataset_names=['afw','lfpw','helen']
    bbox_set={'afw':{}, 'lfpw':{}, 'helen':{}}
    for iset, name in enumerate(files):
        dataset_name=dataset_names[iset]
        file=loadmat('./BoundingBoxes/'+name)['bounding_boxes'][0]
        for info in file:
            img_boxes=list(info[0][0])
            img_name=img_boxes[0][0]
            bboxes=[]
            for box in img_boxes[1:]:
                bboxes.append(box[0])
            bboxes=np.array(bboxes)
            bbox_set[dataset_name][img_name]=bboxes
    return bbox_set

# generate bbox from shape
def GenerateBBox(shape):
    x1=np.min(shape[:, 0])
    y1=np.min(shape[:, 1])
    x2=np.max(shape[:, 0])
    y2=np.max(shape[:, 1])
    return np.array([x1, y1, x2, y2])

# choose the box that contains the shape
def ChooseBox(img, shape, bboxes_set):
    bboxes=bboxes_set[img.split('/')[0]][img.split('/')[-1].strip()]
    for bbox in bboxes:
        flag=True
        for point in shape:
            if point[0]<bbox[0] or point[0]>bbox[2] or point[1]<bbox[1] or point[1]>bbox[3]:
                flag=False
                break
        if flag:
            return bbox

####################### random forest ############################
# from (scipy)random forest extract leave node, get leaves' index
def GetLeaves(forest):
    leaves=[]
    num=0
    for estimator in forest.estimators_:
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()
            if (children_left[node_id] != children_right[node_id]):
                stack.append(children_left[node_id])
                stack.append(children_right[node_id])
            else:
                is_leaves[node_id] = True
        inds=np.where(is_leaves==True)[0]
        num+=len(inds)
        leaves.append(dict(zip(inds, range(len(inds)))))
    return leaves, num

# transform the random positions to landmarks-centerd
def GetLocalFeatureAbsolutePos(poses, bbox, center):
    a_poses=poses.copy()
    w=bbox[2]-bbox[0]
    h=bbox[3]-bbox[1]
    a_poses[:, 0]=a_poses[:, 0]*w+center[0]
    a_poses[:, 1]=a_poses[:, 1]*h+center[1]
    return a_poses

########################## error ###################################
# compute inter-pupil distance normalized landmark error as mentioned in paper
def ComputeError(shapes, gts):
    err=0
    for i in range(len(shapes)):
        shape=shapes[i]
        gt=gts[i]
        pupil_dist=np.sqrt(np.sum(np.square(np.mean(gt[42:48]-gt[36:42], 0))))
        err+=np.sum(np.sqrt(np.sum(np.square(shape-gt), 1)))/(param_landmark_num*pupil_dist)
    return err/len(shapes)

########################### training phase ###########################
def LoadData():
    imgs=[]
    shapes=[]
    bboxes=[]
    img_list=open('./300w_cropped/train_img_list').readlines()
    bboxes_set=LoadBBox()
    i=0
    for img in img_list:
        if True:
            i+=1
            print(i)
            imgs.append(cv2.imread('./300w_cropped/'+img.strip(), 0).astype(np.float))
            #imgs.append('/Users/lrg/face_datasets/300w_cropped/'+img.strip())
            shapes.append(ReadShape('./300w_cropped/'+img.split('.')[0]+'.pts'))
            #bboxes.append(ChooseBox(img, shapes[-1], bboxes_set))
            bboxes.append(GenerateBBox(shapes[-1]))
    imgs=np.array(imgs)
    shapes=np.array(shapes)
    bboxes=np.array(bboxes)
    mean_rshape=np.mean([Shape2Relative(shape, bboxes[i]) for i,shape in enumerate(shapes)], 0)
    return imgs, shapes, bboxes, mean_rshape

# prepare training data, data augmentation, initialize train_shape
def GetTrainData(imgs, shapes, bboxes):
    train_imgs=[]
    train_shapes=[]
    train_bboxes=[]
    gt_shapes=[]
    # for every training sample choose $augment_num other training shape as initialization
    for i in range(len(imgs)):
        aug_inds=[]
        for j in range(param_augment_num):
            aug_ind=i
            while aug_ind==i or aug_ind in aug_inds:
                aug_ind=np.random.randint(len(imgs))
            aug_inds.append(aug_ind)
            train_imgs.append(imgs[i])
            train_shapes.append(Shape2Absolute(Shape2Relative(shapes[aug_ind], bboxes[aug_ind]), bboxes[i]))
            train_bboxes.append(bboxes[i])
            gt_shapes.append(shapes[i])
    train_imgs=np.array(train_imgs)
    train_shapes=np.array(train_shapes)
    train_bboxes=np.array(train_bboxes)
    gt_shapes=np.array(gt_shapes)

    assert train_imgs.shape[0]==train_shapes.shape[0]
    assert train_imgs.shape[0]==train_bboxes.shape[0]
    assert train_imgs.shape[0]==gt_shapes.shape[0]
    # shuffle
    indexs=np.arange(len(train_imgs))
    np.random.shuffle(indexs)
    return train_imgs[indexs], train_shapes[indexs], train_bboxes[indexs], gt_shapes[indexs]

# compute regression target
def GetTarget(train_shapes, train_bboxes, gt_shapes, mean_rshape):
    targets=[]
    for i in range(len(train_shapes)):
        # do similarity transformation to mean space
        sim_trans=transform.estimate_transform('similarity', CenterShape(Shape2Relative(train_shapes[i], train_bboxes[i])), CenterShape(mean_rshape))
        targets.append(sim_trans(Shape2Relative(gt_shapes[i], train_bboxes[i])-Shape2Relative(train_shapes[i], train_bboxes[i])))
    return np.array(targets)

# train random forests for each landmark to get local binary features
def GetBinFeatures(stage, train_imgs, train_shapes, train_bboxes, mean_rshape, targets):
    bin_features=[]
    forests=[]
    random_poses=[]
    for ilandmark in range(param_landmark_num):
        t1=time.time()
        #### get locations
        feature_pair_pos=np.zeros((param_local_feature_num[stage]*2, 2))
        for i in range(param_local_feature_num[stage]):
            while True:
                pair=np.random.rand(4)*2-1
                x1,y1,x2,y2=pair
                if x1*x1+y1*y1<1 and x2*x2+y2*y2<1 and (x1, y1)!=(x2, y2):
                    break
            feature_pair_pos[2*i:2*i+2]=(pair*param_local_radius[stage]).reshape((2,2))
        random_poses.append(feature_pair_pos)
        #### get pixel difference
        features=np.zeros((len(train_shapes), param_local_feature_num[stage]))
        for i in range(len(train_shapes)):
            #origin_img=cv2.imread(train_imgs[i], 0).astype(np.float)
            origin_img=train_imgs[i]
            # transform from mean space to current training space
            sim_trans=transform.estimate_transform('similarity', CenterShape(mean_rshape), CenterShape(Shape2Relative(train_shapes[i], train_bboxes[i])))
            #trans_feature_pair_pos=Shape2Absolute(sim_trans(feature_pair_pos), train_bboxes[i])+train_shapes[i][ilandmark]
            trans_feature_pair_pos=GetLocalFeatureAbsolutePos(sim_trans(feature_pair_pos), train_bboxes[i], train_shapes[i][ilandmark]).astype(np.int)
            #trans_feature_pair_pos=trans_feature_pair_pos.astype(np.int)
            for j in range(param_local_feature_num[stage]):
                x1,y1=trans_feature_pair_pos[2*j]
                x2,y2=trans_feature_pair_pos[2*j+1]
                # in case out of boundary
                x1=max(0, min(origin_img.shape[1]-1, x1))
                x2=max(0, min(origin_img.shape[1]-1, x2))
                y1=max(0, min(origin_img.shape[0]-1, y1))
                y2=max(0, min(origin_img.shape[0]-1, y2))
                features[i,j]=origin_img[y1,x1] - origin_img[y2,x2]
            #del origin_img
            #gc.collect()
        #### train random forest
        forest=RandomForestRegressor(max_depth=param_tree_depth, n_estimators=param_tree_num, n_jobs=8)
        forest.fit(features, targets[:, ilandmark])
        forests.append(forest)
        #### extract binary features for every training sample
        leaves, leaves_num=GetLeaves(forest)
        reach_nodes=forest.apply(features)
        landmark_bin_features=np.zeros((len(train_shapes), leaves_num))
        for i in range(len(train_shapes)):
            begin_leaf_ind=0
            for j in range(len(leaves)):
                node=reach_nodes[i, j]
                landmark_bin_features[i][begin_leaf_ind+leaves[j][node]]=1
                begin_leaf_ind+=len(leaves[j])
        bin_features.append(landmark_bin_features)
        print('landmark:', ilandmark+1, 'use:', time.time()-t1, 's')
    return np.hstack(bin_features), forests, random_poses

# do global regression
def GlobalRegression(local_binary_features, targets):
    t1=time.time()
    updates=np.zeros((len(targets), param_landmark_num, 2))
    svrs=[]
    for i in range(param_landmark_num):
        # dx
        svr_x=LinearSVR(C=1./len(targets), dual=True, loss='squared_epsilon_insensitive', epsilon=0.0001)
        svr_x.fit(local_binary_features, targets[:, i, 0])
        updates[:, i, 0]=svr_x.predict(local_binary_features)
        # dy
        svr_y=LinearSVR(C=1./len(targets), dual=True, loss='squared_epsilon_insensitive', epsilon=0.0001)
        svr_y.fit(local_binary_features, targets[:, i, 1])
        updates[:, i, 1]=svr_y.predict(local_binary_features)
        svrs.append([svr_x, svr_y])
    print('Global Regression use:', time.time()-t1, 's')
    return updates, svrs

# update train_shapes from each stage
def UpdateShape(shapes, updates, mean_rshape, bboxes):
    for i in range(len(shapes)):
        sim_trans=transform.estimate_transform('similarity', CenterShape(mean_rshape), CenterShape(Shape2Relative(shapes[i], bboxes[i])))
        shapes[i]=Shape2Absolute(sim_trans(updates[i]) + Shape2Relative(shapes[i], bboxes[i]), bboxes[i])

######################### test phase ##############################
# load test data
def LoadTestData():
    imgs=[]
    shapes=[]
    bboxes=[]
    test_img_list=open('./300w_cropped/test_img_list').readlines()
    for i, img in enumerate(test_img_list):
        print(i)
        if False:
            break
        imgs.append(cv2.imread('./300w_cropped/'+img.strip(), 0).astype(np.float))
        #imgs.append('/Users/lrg/face_datasets/300w_cropped/'+img.strip())
        shapes.append(ReadShape('./300w_cropped/'+img.split('.')[0]+'.pts'))
        bboxes.append(GenerateBBox(shapes[-1]))
    imgs=np.array(imgs)
    shapes=np.array(shapes)
    bboxes=np.array(bboxes)
    return imgs, shapes, bboxes

# use trained random forests and picked random local points to extract local binary features for landmarks
def TestGetBinFeatures(imgs, shapes, bboxes, mean_rshape, random_poses, forests):
    bin_features=[]
    for ilandmark in range(param_landmark_num):
        t1=time.time()
        feature_pair_pos=random_poses[ilandmark]
        forest=forests[ilandmark]
        features=np.zeros((len(shapes), feature_pair_pos.shape[0]//2))
        for i in range(len(shapes)):
            #origin_img=cv2.imread(imgs[i], 0).astype(np.float)
            origin_img=imgs[i]
            # transform from mean space to current training space
            sim_trans=transform.estimate_transform('similarity', CenterShape(mean_rshape), CenterShape(Shape2Relative(shapes[i], bboxes[i])))
            #trans_feature_pair_pos=Shape2Absolute(sim_trans(feature_pair_pos), train_bboxes[i])+train_shapes[i][ilandmark]
            trans_feature_pair_pos=GetLocalFeatureAbsolutePos(sim_trans(feature_pair_pos), bboxes[i], shapes[i][ilandmark]).astype(np.int)
            #trans_feature_pair_pos=trans_feature_pair_pos.astype(np.int)
            for j in range(feature_pair_pos.shape[0]//2):
                x1,y1=trans_feature_pair_pos[2*j]
                x2,y2=trans_feature_pair_pos[2*j+1]
                # in case out of boundary
                x1=max(0, min(origin_img.shape[1]-1, x1))
                x2=max(0, min(origin_img.shape[1]-1, x2))
                y1=max(0, min(origin_img.shape[0]-1, y1))
                y2=max(0, min(origin_img.shape[0]-1, y2))
                #import pdb;pdb.set_trace()
                features[i,j]=origin_img[y1,x1] - origin_img[y2,x2]
            #del origin_img
            #gc.collect()
        #### extract binary features for every training sample
        leaves, leaves_num=GetLeaves(forest)
        reach_nodes=forest.apply(features)
        landmark_bin_features=np.zeros((len(shapes), leaves_num))
        for i in range(len(shapes)):
            begin_leaf_ind=0
            for j in range(len(leaves)):
                node=reach_nodes[i, j]
                landmark_bin_features[i][begin_leaf_ind+leaves[j][node]]=1
                begin_leaf_ind+=len(leaves[j])
        bin_features.append(landmark_bin_features)
        print('landmark', ilandmark+1, time.time()-t1, 's')
    return np.hstack(bin_features)

# use trained linear models to predict updates for each stage
def TestGlobalRegression(local_binary_features, svrs):
    updates=np.zeros((len(local_binary_features), param_landmark_num, 2))
    for i in range(param_landmark_num):
        svr_x, svr_y=svrs[i]
        # dx
        updates[:, i, 0]=svr_x.predict(local_binary_features)
        # dy
        updates[:, i, 1]=svr_y.predict(local_binary_features)
    return updates


######################### model ##################################
# save trained models
def SaveModels(random_poses, forests, svrs, mean_rshape, stage):
    with open('stage-'+str(stage), 'wb') as f:
        pickle.dump([random_poses, forests, svrs, mean_rshape], f)

# load models for test
def LoadModels(filename):
    with open(filename, 'rb') as f:
        random_poses, forests, svrs, mean_rshape = pickle.load(f)
    return random_poses, forests, svrs, mean_rshape
