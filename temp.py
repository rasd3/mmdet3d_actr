#!/usr/bin/env python
# -*- coding: utf-8 -*-

        if False:
            temp_idx = group_idx[0].reshape(-1)
            count_idx = torch.zeros((20000))
            for i in range(temp_idx.shape[0]):
                count_idx[temp_idx[i]] += 1
            max_num = temp_idx.max()
            print('max_num: %d, nonzero_num: %d' % (temp_idx.max(), count_idx.nonzero().shape[0]))
            print('max_used: %d' % count_idx.max())
            print('mean_used: %f' % (count_idx[:max_num].mean()))

