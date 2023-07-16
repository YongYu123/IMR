import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

class GenerateMask:
    def __init__(self,seed_number=0):
        self.seed_number = seed_number

    def generate_partial_masks(self,num,m_percentage=0,size=(64,64)):
        masks=[]
        tem_random_seed = self.seed_number
        num = int(num)
        for n in range(num):
            mask = torch.ones(size)
            sum_size = mask.sum()
            
            while 1-mask.sum()/sum_size < m_percentage:
                
                random.seed(tem_random_seed)
                miss_start_x=random.randrange(mask.size(0))
                random.seed(tem_random_seed+1)
                miss_start_y=random.randrange(mask.size(1))
                random.seed(tem_random_seed+2)
                miss_width = random.randrange(0, int(mask.size(0)*m_percentage))
                random.seed(tem_random_seed+3)
                miss_height = random.randrange(0, int(mask.size(1)*m_percentage))
                
                miss_end_x = miss_start_x+miss_width if miss_start_x+miss_width < mask.size(0) else mask.size(0)
                miss_end_y = miss_start_y+miss_height if miss_start_y+miss_height < mask.size(1) else mask.size(1)
                mask[miss_start_x:miss_end_x, miss_start_y:miss_end_y] = 0
                tem_random_seed += 1
            masks.append(mask)   
        return masks
    
    def generate_full_masks(self,num,size=(64,64)):
        num = int(num)
        masks=[]
        for n in range(num):
            mask = torch.zeros(size)
            masks.append(mask)   
        return masks
    
    def generate_random_masks(self,num,p_num,f_num,m_percentage=0,size=(64,64)):
        num = int(num)
        p_num = int(p_num)
        f_num = int(f_num)
        masks=[]
        for n in range(num):
            mask = torch.ones(size)
            masks.append(mask)
        partial_masks = self.generate_partial_masks(p_num,m_percentage,size)
        
        full_masks = self.generate_full_masks(f_num,size)
        total_miss_masks = partial_masks+full_masks
        random.seed(self.seed_number)
        random_index = random.sample(range(num), f_num+p_num)
        for i in range(f_num+p_num):
            masks[random_index[i]] = total_miss_masks[i]
        
        return torch.unsqueeze(torch.stack(masks),1)
