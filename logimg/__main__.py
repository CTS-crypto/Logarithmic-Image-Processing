import matplotlib.pyplot as plt
import json
import numpy as np

# region graphics

with open('logimg/torax/contrast.json') as f:
    contrast0 = json.load(f)

with open('logimg/torax/affine_transform/equalize/contrast.json') as f:
    contrast = json.load(f)

with open('logimg/torax/affine_transform/hlip/contrast.json') as f:
    contrast2 = json.load(f)

with open('logimg/torax/affine_transform/slip/contrast.json') as f:
    contrast3 = json.load(f)

# with open('logimg/torax/affine_transform/hlip-/contrast.json') as f:
#     contrast3_5 = json.load(f)

# with open('logimg/torax/affine_transform/pslip/contrast.json') as f:
#     contrast4 = json.load(f)

# with open('logimg/torax/affine_transform/slip/contrast.json') as f:
#     contrast5 = json.load(f)

    

# with open('logimg/torax/affine_transform/plip/contrast_256.json') as f:
#     contrast6 = json.load(f)
    
# with open('logimg/torax/affine_transform/plip/contrast_300.json') as f:
#     contrast7 = json.load(f)

# with open('logimg/torax/affine_transform/plip/contrast_1026.json') as f:
#     contrast8 = json.load(f)

# with open('logimg/torax/affine_transform/plip/contrast_4100.json') as f:
#     contrast9 = json.load(f)



# with open('logimg/torax/affine_transform/ppslip/contrast_256.json') as f:
#     contrast10 = json.load(f)
    
# with open('logimg/torax/affine_transform/ppslip/contrast_300.json') as f:
#     contrast11 = json.load(f)

# with open('logimg/torax/affine_transform/ppslip/contrast_1026.json') as f:
#     contrast12 = json.load(f)

# with open('logimg/torax/affine_transform/ppslip/contrast_4100.json') as f:
#     contrast13 = json.load(f)

    

# data_plip=[]

# for i in contrast9:
#     data_plip.append(max(contrast6[i],contrast7[i],contrast8[i],contrast9[i]))
#     print(contrast6[i],contrast7[i],contrast8[i],contrast9[i])


# data_ppslip=[]

# for i in contrast10:
#     data_ppslip.append(max(contrast10[i],contrast11[i],contrast12[i],contrast13[i]))
#     print(contrast10[i],contrast11[i],contrast12[i],contrast13[i])
    

data=[list(contrast0.values()),list(contrast.values()),list(contrast2.values()),list(contrast3.values())]

plt.boxplot(data,labels=['Original','Ecualizada','HLIP','SLIP'])
plt.show()

#endregion

# with open('logimg/torax/affine_transform/ppslip/contrast_256.json') as f:
#     contrast6 = json.load(f)

# with open('logimg/torax/affine_transform/ppslip/contrast_300.json') as f:
#     contrast7 = json.load(f)    

# with open('logimg/torax/affine_transform/ppslip/contrast_1026.json') as f:
#     contrast8 = json.load(f)

# with open('logimg/torax/affine_transform/ppslip/contrast_4100.json') as f:
#     contrast9 = json.load(f)

# data_plip=[]

# for i in contrast6:
#     data_plip.append(max(contrast6[i],contrast7[i],contrast8[i],contrast9[i]))
#     print(contrast6[i],contrast7[i],contrast8[i],contrast9[i])

# # print(np.mean(contrast),np.mean(contrast2),np.mean(contrast3),np.mean(contrast4),np.mean(contrast5))
# # print(np.median(contrast),np.median(contrast2),np.median(contrast3),np.median(contrast4),np.median(contrast5))

# print(np.mean(data_plip))
# print(np.median(data_plip))