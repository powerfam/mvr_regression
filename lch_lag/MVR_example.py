#!/usr/bin/env python
# coding: utf-8

# # 데이터 프레임 출력

# In[8]:


raw_processed


# # 함수 실행

# In[9]:


import lch_lag.mvr_total_test as mvr
mvr.mvr_findlag(raw_processed, var_number=5, start_lag=6, end_lag=12, resid_test=1, shap_value_test=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


test_code= {'soyoil': '06_02_00_00_01_00_1103_NO', 
            'soybean': '06_01_00_00_01_00_1103_NO',
             'palmoil': '06_02_00_00_02_01_1446_NO',
             'myr_usd': 'MYR_USD',
             'palm_stock': '02_03_03_HB_00_00_MYS_00_84_04',
             'brent': '02_01_01_00_02_00_1301_NO',
            'dollar': '01_02_02_03_05_05_USA_00_251_06'
            }


x = k.KDB(code=test_code, info_print=False)
raw = x.df()



# # brent oil unit change(bbl -> ton)
raw['brent'] = raw['brent']/0.136

# soyoil unit (pound -> ton)
raw['soyoil'] = raw['soyoil']/0.000454/100

# fx adjusting
raw['palmoil'] = raw['palmoil']/raw['myr_usd']

# soyoil-palmoil spread calculation
raw['soy_palm_spread'] = raw['soyoil']-raw['palmoil']

# palm-gasoil spread calculation
raw['palm_brent_spread'] = np.log(np.array(raw['palmoil'])) - np.log(np.array(raw['brent']))



feature = ['soyoil', 'soybean', 'palm_stock', 'soy_palm_spread', 'palm_brent_spread', 'dollar']
raw = raw[feature]


# 모든 변수에 대해서 log 변환
raw['soyoil_log'] = pd.DataFrame(np.log(np.array(raw['soyoil'])), index=raw.index)
raw['soybean_log'] = pd.DataFrame(np.log(np.array(raw['soybean'])), index=raw.index)
raw['palm_stock_log'] = pd.DataFrame(np.log(np.array(raw['palm_stock'])), index=raw.index) 
raw['soy_palm_spread_log'] = pd.DataFrame(np.log(np.array(raw['soy_palm_spread'])), index=raw.index)
# raw['palm_brent_spread_log'] = pd.DataFrame(np.log(np.array(raw['palm_brent_spread'])), index=raw.index)  #마이너스가 있어서 MoM으로 변환함
raw['palm_brent_spread_log'] = raw['palm_brent_spread']
raw['dollar_log'] = pd.DataFrame(np.log(np.array(raw['dollar'])), index=raw.index)
raw_processed = raw.copy()
raw_processed = raw_processed.loc['2005':, 'soyoil_log':'dollar_log']
raw_processed


# In[2]:





# In[ ]:


mvr.mvr_findlag()

