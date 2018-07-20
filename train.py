from FaceAlignment import *

if __name__=='__main__':
    imgs, shapes, bboxes, mean_rshape = LoadData()
    train_imgs, train_shapes, train_bboxes, gt_shapes = GetTrainData(imgs, shapes, bboxes)
    print('Initial Error:', ComputeError(train_shapes, gt_shapes))
    forests=[]
    svrs=[]
    random_poses=[]
    for i in range(param_cascade_num):
        print('Cascade stage:', i+1)
        
        targets = GetTarget(train_shapes, train_bboxes, gt_shapes, mean_rshape)
        
        local_binary_features, forests_per_stage, random_poses_per_stage = GetBinFeatures(i, train_imgs, train_shapes, train_bboxes, mean_rshape, targets)
        forests.append(forests_per_stage)
        random_poses.append(random_poses_per_stage)
        
        updates, svrs_per_stage = GlobalRegression(local_binary_features, targets)
        svrs.append(svrs_per_stage)
        #import pdb;pdb.set_trace()
        
        UpdateShape(train_shapes, updates, mean_rshape, train_bboxes)
        
        SaveModels(random_poses, forests, svrs, mean_rshape, i+1)
        
        print('Stage', i+1, 'Error:', ComputeError(train_shapes, gt_shapes))
        #del targets, updates
