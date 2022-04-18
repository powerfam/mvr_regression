#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import warnings
import lag_pkg.mvr_total_test as mvr


# In[2]:


def mvr_findlag(df, var_number, start_lag, end_lag, summary=0, resid_test=0, shap_value_test=0):
            
    """
    Args:
        df: - 데이터프레임(가공된 상태) 필요
            - 생긴형태는 y factor1 factor2 factor3 ,... 가 되어야함.  
            - 최대 5개의 설명변수에 대해서 실행
        
        var_number: - 설명변수의 갯수를 의미함.
                    - y = x1, x2, x3 형태의 경우 3을 입력
                    
        summary: - regression full table을 출력하고 싶은 경우 1을 입력
        
        start_lag: - 체크하고 싶은 최소 lag 시점
                   - 3시차부터 점검하고 싶은 경우 3 입력
        
        end_lag: - 체크하고 싶은 최대 lag 시점
                 - 12시차까지 점검하고 싶은 경우 12 입력
                    
    
    Returns:
        1) AIC 절대값이 0에 가까운 순서로 시차를 고려한 변수 조합 성능 상위 10개가 출력됨.
        2) 해당 시차를 고려한 regression result 테이블이 출력됨
        3) 잔차진단을 위한 잔차 그래프, ACF, PACF 그래프, ADF, KPSS 테스트 결과가 출력됨. 
        
    """
    rng= range(start_lag, end_lag)
    import statsmodels.api as sm
    
    def stationarity_adf_test(Y_Data, Target_name):
        
        if len(Target_name) == 0:
            
            Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data)[0:4],
                                         index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
            for key, value in sm.tsa.stattools.adfuller(Y_Data)[4].items():
                Stationarity_adf['Critical Value(%s)'%key] = value
                Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data)[5]
                Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
        else:
            Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data[Target_name])[0:4],
                                         index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
            for key, value in sm.tsa.stattools.adfuller(Y_Data[Target_name])[4].items():
                Stationarity_adf['Critical Value(%s)'%key] = value
                Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data[Target_name])[5]
                Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
        return Stationarity_adf

    def stationarity_kpss_test(Y_Data, Target_name):
        if len(Target_name) == 0:
            Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data)[0:3],
                                          index=['Test Statistics', 'p-value', 'Used Lag'])
            for key, value in sm.tsa.stattools.kpss(Y_Data)[3].items():
                Stationarity_kpss['Critical Value(%s)'%key] = value
                Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
        else:
            Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data[Target_name])[0:3],
                                          index=['Test Statistics', 'p-value', 'Used Lag'])
            for key, value in sm.tsa.stattools.kpss(Y_Data[Target_name])[3].items():
                Stationarity_kpss['Critical Value(%s)'%key] = value
                Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
        return Stationarity_kpss  
    
    def factor_name_df(data):
        col_real = list(data.columns[1:var_number+1])
        factor_name = pd.DataFrame()
        name_idx = ['명칭']
        for num in range(1, len(col_real)+1):
            factor_name.loc[0,f'factor{num}'] = col_real[num-1]

        factor_name.index = name_idx 
        return factor_name
    
    
    
    
    # 변수 1개
    if var_number == 1:
        
        AIC = []
        factor1_n = []
        
        for a in rng:
            op_df = df.copy()
            op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
            op_df =op_df.dropna()
            y = op_df.iloc[:, 0]
            X = op_df.loc[:, f'factor1_lag{a}':]
            X = sm.add_constant(X, has_constant='add')
            
            model = sm.OLS(y, X)
            fitted_model = model.fit()
            
            # add regression params number & lag number result 
            AIC.append(fitted_model.aic)
            factor1_n.append(a)
            
        AIC = pd.DataFrame(AIC, columns=['AIC'])
        AIC['abs_AIC'] = abs(AIC['AIC'])
        factor1_n = pd.DataFrame(factor1_n, columns=['factor1_log_n'])
        
        comb_df = AIC.join(factor1_n, how='inner')
        a = comb_df.iloc[0,2]
        print("<최적 시차 조합 bset 10>")
        display(comb_df.head(10))
        print("factor1의 최적 시차는 "+f"{a}"+' 입니다')
        print()
        print()
        
        # 신규 lag 변수로 regression result 출력
        print('\033[37m \033[40m' + "<최적 시차를 적용한 다변량 회귀분석 결과>" + '\033[0m')
        op_df = df.copy()
        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
        op_df =op_df.dropna()
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = sm.add_constant(X, has_constant='add')
            
        model = sm.OLS(y, X)
        fitted_model = model.fit()
        var_index = pd.DataFrame(['factor1',], index=op_df.columns[1:var_number+1], columns=['표시'])
        var_lags = pd.DataFrame([a], index=op_df.columns[1:var_number+1], columns=['최적시차'])
        var_pvalues = pd.DataFrame(fitted_model.pvalues[1:].values, index=op_df.columns[1:var_number+1], columns=['p-values'])
        var_coeff = pd.DataFrame(fitted_model.params[1:].values, index=op_df.columns[1:var_number+1], columns =['coefficient'])
        display(pd.concat([var_index, var_lags, var_pvalues, var_coeff], axis=1))

        

        
        
        
    
    
    
    # 변수 2개
    elif var_number == 2:
        
        AIC = []
        factor1_n = []
        factor2_n = []    
        
        for a in rng:
            for b in rng:
                op_df = df.copy()
                op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
                op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
                op_df =op_df.dropna()
                
                y = op_df.iloc[:, 0]
                X = op_df.loc[:, f'factor1_lag{a}':]
                X = sm.add_constant(X, has_constant='add')

                model = sm.OLS(y, X)
                fitted_model = model.fit()

                # add regression params number & lag number result 
                AIC.append(fitted_model.aic)
                factor1_n.append(a)
                factor2_n.append(b)
                
        AIC = pd.DataFrame(AIC, columns=['AIC'])
        AIC['abs_AIC'] = abs(AIC['AIC'])
        factor1_n = pd.DataFrame(factor1_n, columns=['factor1_log_n'])
        factor2_n = pd.DataFrame(factor2_n, columns=['factor2_log_n'])
        comb_df = AIC.join(factor1_n, how='inner').join(factor2_n, how='inner').sort_values('abs_AIC')
        a = comb_df.iloc[0,2]
        b = comb_df.iloc[0,3]
        print("<최적 시차 조합 bset 10>")
        display(comb_df.head(10))
        print("factor1의 최적 시차는 "+f"{a}"+' 입니다')
        print("factor2의 최적 시차는 "+f"{b}"+' 입니다')
        print()
        print()
        
        # 신규 lag 변수로 regression result 출력
        print('\033[37m \033[40m' + "<최적 시차를 적용한 다변량 회귀분석 결과>" + '\033[0m')
        op_df = df.copy()
        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
        op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
        op_df =op_df.dropna()
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = sm.add_constant(X, has_constant='add')
            
        model = sm.OLS(y, X)
        fitted_model = model.fit()
        var_index = pd.DataFrame(['factor1', 'factor2',], index=op_df.columns[1:var_number+1], columns=['표시'])
        var_lags = pd.DataFrame([a, b], index=op_df.columns[1:var_number+1], columns=['최적시차'])
        var_pvalues = pd.DataFrame(fitted_model.pvalues[1:].values, index=op_df.columns[1:var_number+1], columns=['p-values'])
        var_coeff = pd.DataFrame(fitted_model.params[1:].values, index=op_df.columns[1:var_number+1], columns =['coefficient'])
        display(pd.concat([var_index, var_lags, var_pvalues, var_coeff], axis=1))

        

                
    
    # 변수 3개
    elif var_number == 3:
        AIC = []
        factor1_n = []
        factor2_n = []
        factor3_n = []
        for a in rng:
            for b in rng:
                for c in rng:
                    op_df = df.copy()
                    
                    op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
                    op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
                    op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
                    op_df =op_df.dropna()
                
                    y = op_df.iloc[:, 0]
                    X = op_df.loc[:, f'factor1_lag{a}':]
                    X = sm.add_constant(X, has_constant='add')

                    model = sm.OLS(y, X)
                    fitted_model = model.fit()

                    # add regression params number & lag number result 
                    AIC.append(fitted_model.aic)
                    factor1_n.append(a)
                    factor2_n.append(b)
                    factor3_n.append(c)
        
        AIC = pd.DataFrame(AIC, columns=['AIC'])
        AIC['abs_AIC'] = abs(AIC['AIC'])
        factor1_n = pd.DataFrame(factor1_n, columns=['factor1_log_n'])
        factor2_n = pd.DataFrame(factor2_n, columns=['factor2_log_n'])
        factor3_n = pd.DataFrame(factor3_n, columns=['factor3_log_n'])
        comb_df = AIC.join(factor1_n, how='inner').join(factor2_n, how='inner').join(factor3_n, how='inner').sort_values('abs_AIC')
        
        a = comb_df.iloc[0,2]
        b = comb_df.iloc[0,3]
        c = comb_df.iloc[0,4]
        print("<최적 시차 조합 bset 10>")
        display(comb_df.head(10))
        print("factor1의 최적 시차는 "+f"{a}"+' 입니다')
        print("factor2의 최적 시차는 "+f"{b}"+' 입니다')
        print("factor3의 최적 시차는 "+f"{c}"+' 입니다')
        print()
        print()
        
        # 신규 lag 변수로 regression result 출력
        print('\033[37m \033[40m' + "<최적 시차를 적용한 다변량 회귀분석 결과>" + '\033[0m')
        op_df = df.copy()
        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
        op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
        op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
        op_df =op_df.dropna()
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = sm.add_constant(X, has_constant='add')
            
        model = sm.OLS(y, X)
        fitted_model = model.fit()
        var_index = pd.DataFrame(['factor1', 'factor2', 'factor3'], index=op_df.columns[1:var_number+1], columns=['표시'])
        var_lags = pd.DataFrame([a, b, c], index=op_df.columns[1:var_number+1], columns=['최적시차'])
        var_pvalues = pd.DataFrame(fitted_model.pvalues[1:].values, index=op_df.columns[1:var_number+1], columns=['p-values'])
        var_coeff = pd.DataFrame(fitted_model.params[1:].values, index=op_df.columns[1:var_number+1], columns =['coefficient'])
        display(pd.concat([var_index, var_lags, var_pvalues, var_coeff], axis=1))

        

    
    # 변수 4개
    elif var_number == 4:
        AIC = []
        factor1_n = []
        factor2_n = []
        factor3_n = []
        factor4_n = []
        for a in rng:
            for b in rng:
                for c in rng:
                    for d in rng:
                        op_df = df.copy()
                        
                        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
                        op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
                        op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
                        op_df[f'factor4_lag{d}'] = df.iloc[:, 4].shift(d)
                        op_df =op_df.dropna()

                        y = op_df.iloc[:, 0]
                        X = op_df.loc[:, f'factor1_lag{a}':]
                        X = sm.add_constant(X, has_constant='add')

                        model = sm.OLS(y, X)
                        fitted_model = model.fit()

                        # add regression params number & lag number result 
                        AIC.append(fitted_model.aic)
                        factor1_n.append(a)
                        factor2_n.append(b)
                        factor3_n.append(c)
                        factor4_n.append(d)
                        
        AIC = pd.DataFrame(AIC, columns=['AIC'])
        AIC['abs_AIC'] = abs(AIC['AIC'])
        factor1_n = pd.DataFrame(factor1_n, columns=['factor1_log_n'])
        factor2_n = pd.DataFrame(factor2_n, columns=['factor2_log_n'])
        factor3_n = pd.DataFrame(factor3_n, columns=['factor3_log_n'])
        factor4_n = pd.DataFrame(factor4_n, columns=['factor4_log_n'])
        comb_df = AIC.join(factor1_n, how='inner').join(factor2_n, how='inner').join(factor3_n, how='inner').join(factor4_n, how='inner').sort_values('abs_AIC')                 
                        
                        
        a = comb_df.iloc[0,2]
        b = comb_df.iloc[0,3]
        c = comb_df.iloc[0,4]           
        d = comb_df.iloc[0,5]
        print("<최적 시차 조합 bset 10>")
        display(comb_df.head(10))
        print("factor1의 최적 시차는 "+f"{a}"+' 입니다')
        print("factor2의 최적 시차는 "+f"{b}"+' 입니다')
        print("factor3의 최적 시차는 "+f"{c}"+' 입니다')
        print("factor4의 최적 시차는 "+f"{d}"+' 입니다')
        print()
        print()
        
        # 신규 lag 변수로 regression result 출력
        print('\033[37m \033[40m' + "<최적 시차를 적용한 다변량 회귀분석 결과>" + '\033[0m')
        op_df = df.copy()
        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
        op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
        op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
        op_df[f'factor4_lag{d}'] = df.iloc[:, 4].shift(d)
        op_df =op_df.dropna()
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = sm.add_constant(X, has_constant='add')  
        
        model = sm.OLS(y, X)
        fitted_model = model.fit()
        var_index = pd.DataFrame(['factor1', 'factor2', 'factor3', 'factor4'], index=op_df.columns[1:var_number+1], columns=['표시'])
        var_lags = pd.DataFrame([a, b, c, d], index=op_df.columns[1:var_number+1], columns=['최적시차'])
        var_pvalues = pd.DataFrame(fitted_model.pvalues[1:].values, index=op_df.columns[1:var_number+1], columns=['p-values'])
        var_coeff = pd.DataFrame(fitted_model.params[1:].values, index=op_df.columns[1:var_number+1], columns =['coefficient'])
        display(pd.concat([var_index, var_lags, var_pvalues, var_coeff], axis=1))
        
       
        
        
        

    # 변수 5개
    elif var_number == 5 :
        AIC = []
        factor1_n = []
        factor2_n = []
        factor3_n = []
        factor4_n = []
        factor5_n = []
        for a in rng:
            for b in rng:
                for c in rng:
                    for d in rng:
                        for e in rng:
                            op_df = df.copy()
                            
                            op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
                            op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
                            op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
                            op_df[f'factor4_lag{d}'] = df.iloc[:, 4].shift(d)
                            op_df[f'factor5_lag{e}'] = df.iloc[:, 5].shift(e)
                            op_df =op_df.dropna()

                            y = op_df.iloc[:, 0]
                            X = op_df.loc[:, f'factor1_lag{a}':]
                            X = sm.add_constant(X, has_constant='add')

                            model = sm.OLS(y, X)
                            fitted_model = model.fit()

                            # add regression params number & lag number result 
                            AIC.append(fitted_model.aic)
                            factor1_n.append(a)
                            factor2_n.append(b)
                            factor3_n.append(c)
                            factor4_n.append(d)
                            factor5_n.append(e)
                            
        AIC = pd.DataFrame(AIC, columns=['AIC'])
        AIC['abs_AIC'] = abs(AIC['AIC'])
        factor1_n = pd.DataFrame(factor1_n, columns=['factor1_log_n'])
        factor2_n = pd.DataFrame(factor2_n, columns=['factor2_log_n'])
        factor3_n = pd.DataFrame(factor3_n, columns=['factor3_log_n'])
        factor4_n = pd.DataFrame(factor4_n, columns=['factor4_log_n'])
        factor5_n = pd.DataFrame(factor5_n, columns=['factor5_log_n'])
        comb_df = AIC.join(factor1_n, how='inner').join(factor2_n, how='inner').join(factor3_n, how='inner').join(factor4_n, how='inner').join(factor5_n, how='inner').sort_values('abs_AIC')                    
                        
                        
        a = comb_df.iloc[0,2]
        b = comb_df.iloc[0,3]
        c = comb_df.iloc[0,4]           
        d = comb_df.iloc[0,5]
        e = comb_df.iloc[0,6]
        print("<최적 시차 조합 bset 10>")
        display(comb_df.head(10))
        print("factor1의 최적 시차는 "+f"{a}"+' 입니다')
        print("factor2의 최적 시차는 "+f"{b}"+' 입니다')
        print("factor3의 최적 시차는 "+f"{c}"+' 입니다')
        print("factor4의 최적 시차는 "+f"{d}"+' 입니다')
        print("factor5의 최적 시차는 "+f"{e}"+' 입니다')
        print()
        print()
        
        # 신규 lag 변수로 regression result 출력
        print('\033[37m \033[40m' + "<최적 시차를 적용한 다변량 회귀분석 결과>" + '\033[0m')
#         print("<최적 시차를 적용한 다변량 회귀분석 결과>")
        op_df = df.copy()
        op_df[f'factor1_lag{a}'] = df.iloc[:, 1].shift(a)
        op_df[f'factor2_lag{b}'] = df.iloc[:, 2].shift(b)
        op_df[f'factor3_lag{c}'] = df.iloc[:, 3].shift(c)
        op_df[f'factor4_lag{d}'] = df.iloc[:, 4].shift(d)
        op_df[f'factor5_lag{e}'] = df.iloc[:, 5].shift(e)
        op_df =op_df.dropna()
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = sm.add_constant(X, has_constant='add')
            
        model = sm.OLS(y, X)
        fitted_model = model.fit()
        var_index = pd.DataFrame(['factor1', 'factor2', 'factor3', 'factor4', 'factor5'], index=op_df.columns[1:var_number+1], columns=['표시'])
        var_lags = pd.DataFrame([a, b, c, d, e], index=op_df.columns[1:var_number+1], columns=['최적시차'])
        var_pvalues = pd.DataFrame(fitted_model.pvalues[1:].values, index=op_df.columns[1:var_number+1], columns=['p-values'])
        var_coeff = pd.DataFrame(fitted_model.params[1:].values, index=op_df.columns[1:var_number+1], columns =['coefficient'])
        display(pd.concat([var_index, var_lags, var_pvalues, var_coeff], axis=1))
        
        
        
    if summary == 1:
        display(fitted_model.summary())
            
    if resid_test == 1:
        #잔차 진단
        print('\033[37m \033[40m' + "<잔차 진단 결과물>" + '\033[0m')
        print()
        print()
        print('<잔차 확인>')
        fitted_model.resid.plot()
        plt.show()


        print()
        print()
        print('잔차 ACF/PACF')
        sm.graphics.tsa.plot_acf(fitted_model.resid, lags=24, ax=plt.subplot(211))
        plt.xlim(-1, 24)
        plt.ylim(-1.1, 1.1)
        plt.title("Residual ACF")


        sm.graphics.tsa.plot_pacf(fitted_model.resid, lags=24, ax=plt.subplot(212))
        plt.xlim(-1, 24)
        plt.ylim(-1.1, 1.1)
        plt.title("Residual PACF")
        plt.tight_layout()
        plt.show()

        print()
        adf_dic = {'ADF test H0': '비정상 상태/시간의존 구조', 'H0 기각 O':'정상성 확보', 'H0 기각 X':'비정상 상태'}
        display(pd.DataFrame(adf_dic.values(), index=adf_dic.keys(), columns=['설명']))
        display(stationarity_adf_test(fitted_model.resid, []))

        print()
        kpss_dic = {'KPSS test H0': '정상 상태/시간 비의존 구조', 'H0 기각 O':'비정상성 상태', 'H0 기각 X':'정상성 확보'}
        display(pd.DataFrame(kpss_dic.values(), index=kpss_dic.keys(), columns=['설명']))
        display(stationarity_kpss_test(fitted_model.resid, []))
        print()
        print()
        print()
    
    if shap_value_test == 1:
        import shap
        from sklearn.linear_model import LinearRegression
        print()
        print()
        print('\033[37m \033[40m' + "<Shap Value를 활용한 변수 중요도 분석>" + '\033[0m')
        y = op_df.iloc[:, 0]
        X = op_df.loc[:, f'factor1_lag{a}':]
        X = X.iloc[:, 0:var_number]
        model = LinearRegression()
        model.fit(X, y)
        explainer = shap.LinearExplainer(model, X) # 모델 Shap Value 계산 객체 지정
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)
        print("그래프 상 특성은 예측에 미치는 영향력(중요도)에 따라서 정렬됨")
        print("예측에 영향을 미치는 순서는 위에서 순서대로")
        
        display(factor_name_df(df))
       

           
      


# In[ ]:




