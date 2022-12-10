import matplotlib.pyplot as plt
import json
import numpy as np

# region graphics

# with open('LogImgPro/sum/medicas/contrast.json') as f:
#     contrast0 = json.load(f)

with open('LogImgPro/sum/medicas/l_sum.json') as f:
    contrast = json.load(f)

with open('LogImgPro/sum/medicas/lip_sum.json') as f:
    contrast2 = json.load(f)

with open('LogImgPro/sum/medicas/hlip_sum.json') as f:
    contrast3 = json.load(f)

# with open('LogImgPro/sum/medicas/hlip-/contrast.json') as f:
#     contrast3_5 = json.load(f)

with open('LogImgPro/sum/medicas/pslip_sum.json') as f:
    contrast4 = json.load(f)

with open('LogImgPro/sum/medicas/slip_sum.json') as f:
    contrast5 = json.load(f)

    

with open('LogImgPro/sum/medicas/plip_sum_256.json') as f:
    contrast6 = json.load(f)
    
with open('LogImgPro/sum/medicas/plip_sum_300.json') as f:
    contrast7 = json.load(f)

with open('LogImgPro/sum/medicas/plip_sum_1026.json') as f:
    contrast8 = json.load(f)

with open('LogImgPro/sum/medicas/plip_sum_4100.json') as f:
    contrast9 = json.load(f)



with open('LogImgPro/sum/medicas/ppslip_sum_256.json') as f:
    contrast10 = json.load(f)
    
with open('LogImgPro/sum/medicas/ppslip_sum_300.json') as f:
    contrast11 = json.load(f)

with open('LogImgPro/sum/medicas/ppslip_sum_1026.json') as f:
    contrast12 = json.load(f)

with open('LogImgPro/sum/medicas/ppslip_sum_4100.json') as f:
    contrast13 = json.load(f)

    

data_plip=[]

for i in contrast6:
    p_1=contrast6[i]
    p_2=contrast7[i.replace('256','300')]
    p_3=contrast8[i.replace('256','1026')]
    p_4=contrast9[i.replace('256','4100')]
    data_plip.append(max(p_1,p_2,p_3,p_4))
    # print(p_1,p_2,p_3,p_4)

data_ppslip=[]

for i in contrast10:
    p_1=contrast10[i]
    p_2=contrast11[i.replace('256','300')]
    p_3=contrast12[i.replace('256','1026')]
    p_4=contrast13[i.replace('256','4100')]
    data_ppslip.append(max(p_1,p_2,p_3,p_4))
    # print(p_1,p_2,p_3,p_4)
    

# data=[list(contrast.values()),list(contrast2.values()),list(contrast3.values()),list(contrast4.values()),list(contrast5.values()),data_plip,data_ppslip]

# plt.boxplot(data,labels=['Lineal','LIP','HLIP','PSLIP','SLIP','PLIP','PPSLIP'])
# plt.show()

#endregion

# with open('LogImgPro/sum/medicas/ppslip/contrast_256.json') as f:
#     contrast6 = json.load(f)

# with open('LogImgPro/sum/medicas/ppslip/contrast_300.json') as f:
#     contrast7 = json.load(f)    

# with open('LogImgPro/sum/medicas/ppslip/contrast_1026.json') as f:
#     contrast8 = json.load(f)

# with open('LogImgPro/sum/medicas/ppslip/contrast_4100.json') as f:
#     contrast9 = json.load(f)

# data_plip=[]

# for i in contrast6:
#     data_plip.append(max(contrast6[i],contrast7[i],contrast8[i],contrast9[i]))
#     print(contrast6[i],contrast7[i],contrast8[i],contrast9[i])

print(np.mean(list(contrast.values())),np.mean(list(contrast2.values())),np.mean(list(contrast3.values())),np.mean(list(contrast4.values())),np.mean(list(contrast5.values())))
print(np.median(list(contrast.values())),np.median(list(contrast2.values())),np.median(list(contrast3.values())),np.median(list(contrast4.values())),np.median(list(contrast5.values())))

print(np.mean(data_plip),np.mean(data_ppslip))
print(np.median(data_plip),np.median(data_ppslip))