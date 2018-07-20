from FaceAlignment import *

if __name__=='__main__':
    imgs, gt_shapes, bboxes=LoadTestData()
    random_poses, forests, svrs, mean_rshape=LoadModels('stage-'+str(param_cascade_num))
    test_shapes=np.array([Shape2Absolute(mean_rshape, bboxes[i]) for i in range(len(imgs))])
    print('Initial Error:', ComputeError(test_shapes, gt_shapes))
    for i in range(param_cascade_num):
        t1=time.time()
        
        local_binary_features=TestGetBinFeatures(imgs, test_shapes, bboxes, mean_rshape, random_poses[i], forests[i])        
        t2=time.time()
        print('get binary features:', t2-t1, 's')
        
        updates=TestGlobalRegression(local_binary_features, svrs[i])
        t3=time.time()
        print('global regression:', t3-t2, 's')
        
        UpdateShape(test_shapes, updates, mean_rshape, bboxes)
        print('Stage', i+1, 'Error:', ComputeError(test_shapes, gt_shapes))

