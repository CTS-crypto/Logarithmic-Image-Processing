import matplotlib.pyplot as plt
import json
import numpy as np

# region graphics

with open('logimg/torax/contrast.json') as f:
    contrast0 = json.load(f)

# with open('logimg/sum/l_sum.json') as f:
#     contrast = json.load(f)

# with open('logimg/sum/lip_sum.json') as f:
#     contrast2 = json.load(f)

# with open('logimg/sum/hlip_sum.json') as f:
#     contrast3 = json.load(f)

# with open('logimg/natural/affine_transform/hlip-/contrast.json') as f:
#     contrast3_5 = json.load(f)

# with open('logimg/sum/pslip_sum.json') as f:
#     contrast4 = json.load(f)

# with open('logimg/sum/slip_sum.json') as f:
#     contrast5 = json.load(f)

    

# with open('logimg/sum/plip_sum_256.json') as f:
#     contrast6 = json.load(f)
    
# with open('logimg/sum/plip_sum_300.json') as f:
#     contrast7 = json.load(f)

# with open('logimg/sum/plip_sum_1026.json') as f:
#     contrast8 = json.load(f)

# with open('logimg/sum/plip_sum_4100.json') as f:
#     contrast9 = json.load(f)



# with open('logimg/sum/ppslip_sum_256.json') as f:
#     contrast10 = json.load(f)
    
# with open('logimg/sum/ppslip_sum_300.json') as f:
#     contrast11 = json.load(f)

# with open('logimg/sum/ppslip_sum_1026.json') as f:
#     contrast12 = json.load(f)

# with open('logimg/sum/ppslip_sum_4100.json') as f:
#     contrast13 = json.load(f)

    

# data_plip=[]

# for i in range(len(contrast9)):
#     data_plip.append(max(contrast6[i],contrast7[i],contrast8[i],contrast9[i]))
#     print(contrast6[i],contrast7[i],contrast8[i],contrast9[i])


# data_ppslip=[]

# for i in range(len(contrast10)):
#     data_ppslip.append(max(contrast10[i],contrast11[i],contrast12[i],contrast13[i]))
#     print(contrast10[i],contrast11[i],contrast12[i],contrast13[i])
    

# data=[contrast,contrast2,contrast3,contrast4,contrast5,data_plip,data_ppslip]

# plt.boxplot(data,labels=['Lineal','LIP','HLIP','PSLIP', 'SLIP','PLIP','PPSLIP'])
# plt.show()

#endregion

with open('logimg/natural/affine_transform/equalize/contrast.json') as f:
    contrast6 = json.load(f)

with open('logimg/natural/affine_transform/hlip/contrast.json') as f:
    contrast7 = json.load(f)    

with open('logimg/natural/affine_transform/slip/contrast.json') as f:
    contrast8 = json.load(f)

# with open('logimg/natural/affine_transform/plip/contrast_4100.json') as f:
#     contrast9 = json.load(f)

# data_plip=[]

# for i in contrast6:
#     data_plip.append(max(contrast6[i],contrast7[i],contrast8[i],contrast9[i]))
#     print(contrast6[i],contrast7[i],contrast8[i],contrast9[i])

print(np.mean(list(contrast6.values())),np.mean(list(contrast7.values())),np.mean(list(contrast8.values())))
print(np.median(list(contrast6.values())),np.mean(list(contrast7.values())),np.mean(list(contrast8.values())))