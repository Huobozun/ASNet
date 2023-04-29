import os 

# with open('/home/wpf/zjg/ASnet/filenames/framesampleTRAIN3.txt','w') as f:
    
#     for i in range(0,2644):
        
#         f.writelines('TRAIN/frameleft/{}.pfm TRAIN/frameright/{}.pfm\n'.format(i,i)) 
# with open('/home/wpf/zjg/ASnet/filenames/framesampleTEST3.txt','w') as f:
    
#     for i in range(0,1342):
        
#         f.writelines('TEST/frameleft/{}.pfm TEST/frameright/{}.pfm\n'.format(i,i)) 
with open('/home/wpf/zjg/ASnet/filenames/framesample.txt','w') as f:
    
    for i in range(0,1399):
        
        f.writelines('frameleft/{}.pfm frameright/{}.pfm\n'.format(i,i)) 


# with open('/home/wpf/zjg/ASnet/filenames/frame_train3.txt','w') as f:
    
#     for i in range(0,2644):
        
#         f.writelines('TRAIN/frameleft/{}.pfm TRAIN/frameright/{}.pfm TRAIN/depthleft/{}.pfm\n'.format(i,i,i))

# with open('/home/wpf/zjg/ASnet/filenames/frame_val3.txt','w') as f:
    
#     for i in range(0,200):
        
#         f.writelines('VALID/frameleft/{}.pfm VALID/frameright/{}.pfm VALID/depthleft/{}.pfm\n'.format(i,i,i))