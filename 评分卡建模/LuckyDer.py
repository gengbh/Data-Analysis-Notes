#正则，输入正则表达式，及列名列表，返回对应的列名
import numpy as np
import pandas as pd
def reg(rex,col_list):
    """
    说明：正则表达式，查询符合正则的字段
    """
    import re
    a = [i for i in list(col_list) if re.match(rex,i) != None]
    return a
def iv_xy(x, y):
    # good bad func
    def goodbad(df):
        names = {'good': np.sum(df['y']==0),'bad': np.sum(df['y']==1)}
        return pd.Series(names)
    # iv calculation
    iv_total = pd.DataFrame({'x':x.astype('str'),'y':y}) \
      .fillna('missing') \
      .groupby('x') \
      .apply(goodbad) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv.sum()
    # return iv
    return iv_total
def iv(dat, y):
    # x variable names
    xs = [i for i in dat.columns if i not in [y]]
    # info_value
    a = []
    for i in xs:
        a.append(iv_xy(dat[i], dat[y]))
        print(i)
        print("还剩变量："+ str(len(xs) - len(a)))
    ivlist = pd.DataFrame({
        'variable': xs,
        'info_value': a
    }, columns=['variable', 'info_value'])
    #[iv_xy(dat[i], dat[y]) for i in xs]
    # sorting iv 
    ivlist = ivlist.sort_values(by='info_value', ascending=False)
    return ivlist
def psi(train,test,psi_comb):
    """
    说明：查看训练与测试集之前的psi。并返回psi大于等于psi_comb的变量。
    
    输入：
        train：训练集
        test: 测试集
        psi_comb：psi阈值
    
    返回：
        psi_df： 变量PSI数据集
        drop_psi: psi大于阈值的变量 
    """    
    PSI = {}
    for col in train.columns:
        train_prop = train[col].value_counts(normalize = True, dropna = False)
        test_prop = test[col].value_counts(normalize = True, dropna = False)
        psi = np.sum((train_prop - test_prop) * np.log(train_prop/test_prop))
        PSI[col] = [psi]
    psi_df = pd.DataFrame(PSI,columns = PSI.keys()).T
    psi_df.columns = ['psi']
    drop_psi = list(psi_df[psi_df.psi >= psi_comb].index)
    keep_psi = list(psi_df[psi_df.psi < psi_comb].index)
    #import toad
    #psi_df= toad.metrics.PSI(train,test).sort_values(0).reset_index().rename(columns = {'index':'feature',0:'psi'})
    #drop_psi = list(psi_df[psi_df.psi >= psi_comb].feature)
    return psi_df,drop_psi,keep_psi
def mannual_breaks(pr_bins,data,ylabel='flagy'):
    '''INPUT:
           pr_bins: bins generated through auto-binning process
           data: data in original form
           ylabel: name of ylabel, in str form
       OUTPUT:
           all_breaklist
           droplist: variables that cannot be binned properly
    '''
    
    import matplotlib.pyplot as plt
    xname=list(pr_bins.keys())
    i=0
    all_breaklist={}
    droplist=[]
    while i <(len(pr_bins)):
        b=xname[i]
        p1=sc.woebin_plot(pr_bins[b])
        plt.show(p1)
        print(i,'/',len(pr_bins),'current splitting points:',list(pr_bins[b]['breaks']))
        print(('Adjust breaks for (%s)?\n 1.next\n 2.yes\n 3.back \n 4.drop \n 5.quit'%b)
              )
        #if_adj=input('1:next    2. yes    3.back  4.drop  5.quit\n')
        if_adj=input()
        new_breaks={}
        while if_adj=='2':
            print('please enter the new breaks:')
            try:
                new_breaks_points=input().split(',')       
                new_breaks={b:[float(x) for x in new_breaks_points]}
            
                bins_adj_temp=sc.woebin(data.loc[:,[b,ylabel]], y = ylabel, breaks_list = new_breaks)
                p2=sc.woebin_plot(bins_adj_temp)
                plt.show(p2)
                print('current splitting points:',list(bins_adj_temp[b]['breaks']))
                print(('Adjust breaks for (%s)?\n 1.next\n 2.yes\n 3.back \n 4.drop \n 5.quit'%b)) 
                if_adj = input()
            except: 
                print('Error while adjusting',b)
                if_continue=input('to continue? \n 1.yes 2.next 3.quit')
                if if_continue=='1':
                    continue
                elif if_continue=='2':
                    if_adj=='1'
                elif if_continue=='3':
                    break
        if if_adj=='1':
            if b not in new_breaks.keys():
                all_breaklist[b]=list(pr_bins[b]['breaks'])
            else:
                all_breaklist[b]=new_breaks[b]
            i+=1
            continue
    
        if if_adj=='3':
            if i==0:
                print('This is the first plot, "back" option forbidden')
                break
            else:
                i-=1
        if if_adj=='4':
            droplist.append(b)
            i+=1
        if if_adj=='5':
            break
    print('Mannual adjustment completed')
    return all_breaklist,droplist
#stepwise，基于p值筛选
def stepwise_selection(X, y, 
                       initial_list = [], 
                       threshold_in = 0.01, 
                       threshold_out = 0.05, 
                       verbose = True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    import statsmodels.api as sm
    included = list(initial_list)
    while True:
        changed = False
        # forward step，向前筛选,每次向其中增加一个p值最小的变量。
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        #print(new_pval)
        for new_column in excluded:
            model = sm.GLM(y, sm.add_constant(pd.DataFrame(X[included+[new_column]])), family = sm.families.Binomial()).fit()
            #print(model.pvalues)
            new_pval[new_column] = model.pvalues[new_column]#循环计算每个变量的p值
        print('############################')
        #print(new_pval)
        best_pval = new_pval.min()#p值最小变量
       # print('new_pval:{}\n '.format(new_pval))
        if best_pval < threshold_in:
            best_feature = list(new_pval.index[list(new_pval == best_pval)])[0]
            print('best_feature:{}\n'.format(best_feature))
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        # backward step 向后筛选，每次剔除一个p值最高变量
        model = sm.GLM(y, sm.add_constant(pd.DataFrame(X[included])), family = sm.families.Binomial()).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        print(pvalues)
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = list(pvalues.index[list(pvalues == worst_pval)])[0]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

#计算概率
def predict(model,dat):
    coef_names = model.params.index.tolist()[1:]#变量系数名
    coef_values = model.params.tolist()[1:]#变量系数值
    const = model.params[0] #截距项
    dat = dat[model.params.index.tolist()[1:]]
    x = np.sum(np.array(coef_values) * np.array(dat),axis = 1) + const #计算x
    prob = 1 / (1 + np.exp(-x))
    return prob

def cate_var_transform(X,Y):
    ##取出数据类型
    d_type = X.dtypes #转成一列series
    object_var = X.iloc[:,np.where(d_type == "object")[0]]#筛选出字符型的列，
    num_var = X.iloc[:,np.where(d_type != "object")[0]]#筛选出数值型的列
    
    #object_transfer_rule用于记录每个类别变量的数值转换规则
    object_transfer_rule = list(np.zeros([len(object_var.columns)])) 
    
    #object_transform是类别变量数值化转化后的值
    object_transform = pd.DataFrame(np.zeros(object_var.shape),
                                    columns=object_var.columns) 
    
    for i in range(0,len(object_var.columns)):
        
        temp_var = object_var.iloc[:,i]
        
        ##除空值外的取值种类
        unique_value=np.unique(temp_var.iloc[np.where(~temp_var.isna() )[0]])
    
        transform_rule=pd.concat([pd.DataFrame(unique_value,columns=['raw data']),
                                       pd.DataFrame(np.zeros([len(unique_value),2]),
                                                    columns=['transform data','bad rate'])],axis=1) 
        for j in range(0,len(unique_value)):
            bad_num=len(np.where( (Y == 1) & (temp_var == unique_value[j]) )[0])
            all_num=len(np.where(temp_var == unique_value[j])[0])
            
            #计算badprob
            if all_num == 0:#防止all_num=0的情况，报错
                all_num=0.5  
            transform_rule.iloc[j,2] = 1.0000000*bad_num/all_num
        
        #按照badprob排序，给出转换后的数值
        transform_rule = transform_rule.sort_values(by='bad rate')
        transform_rule.iloc[:,1]=list(range(len(unique_value),0,-1))
         
        #保存转换规则
        object_transfer_rule[i] = transform_rule
        #转换变量
        for k in range(0,len(unique_value)):
            transfer_value = transform_rule.iloc[np.where(transform_rule.iloc[:,0] == unique_value[k])[0],1]
            object_transform.iloc[np.where(temp_var == unique_value[k])[0],i] = float(transfer_value)
        object_transform.iloc[np.where(object_transform.iloc[:,i] == 0)[0],i] = np.nan 
    
    X_transformed = pd.concat([num_var,object_transform],axis = 1) 
    return(X_transformed,object_transfer_rule)
#缺失率
def miss(dat):
    ms = pd.DataFrame(np.round(np.sum(pd.isnull(dat))/len(dat),decimals = 2))
    ms.columns = ['miss_rate']
    return ms




# 生成评分卡
def woepoints_ply1(dtx, binx, x_i, woe_points):
    '''
    Transform original values into woe or porints for one variable.
    
    Params
    ------
    
    Returns
    ------
    
    '''
    # woe_points: "woe" "points"
    # binx = bins.loc[lambda x: x.variable == x_i] 
    # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
    from pandas.api.types import is_numeric_dtype  
    import re
    from pandas.api.types import is_string_dtype
    #每个变量的分箱和分数
    binx = pd.merge(
      binx[['bin']].assign(v1=binx['bin'].str.split('%,%')).explode('v1'), ##explode这步不知道干嘛，没啥变化
      binx[['bin', woe_points]],
      how='left', on='bin'
    ).rename(columns={'v1':'V1',woe_points:'V2'})
    # dtx
    ## cut numeric variable
    if is_numeric_dtype(dtx[x_i]):
        is_sv = pd.Series(not bool(re.search(r'\[', str(i))) for i in binx.V1)
        #print(is_sv)
        binx_sv = binx.loc[is_sv]
        binx_other = binx.loc[~is_sv]
        # create bin column 解析分箱结果
        breaks_binx_other = np.unique(list(map(float, ['-inf']+[re.match(r'.*\[(.*),.+\).*', str(i)).group(1) for i in binx_other['bin']]+['inf'])))
        #print(breaks_binx_other) [-inf 350. 360. 600. 650. 710. 760.  inf]
        labels = ['[{},{})'.format(breaks_binx_other[i], breaks_binx_other[i+1]) for i in range(len(breaks_binx_other)-1)]
        #print(labels) 创建标签
        dtx = dtx.assign(xi_bin = lambda x: pd.cut(x[x_i], breaks_binx_other, right=False, labels=labels))          .assign(xi_bin = lambda x: [i if (i != i) else str(i) for i in x['xi_bin']])
        #print(dtx) 对每个数据打标签，看数据在哪个区间。 column : 变量score结果，变量分箱值
        mask = dtx[x_i].isin(binx_sv['V1'])#取值都是False
        #print(mask.value_counts())
        dtx.loc[mask,'xi_bin'] = dtx.loc[mask, x_i].astype(str)#都是False没法执行
        dtx = dtx[['xi_bin']].rename(columns={'xi_bin':x_i})
    ## to charcarter, na to missing
    if not is_string_dtype(dtx[x_i]):
        dtx.loc[:,x_i] = dtx.loc[:,x_i].astype(str).replace('nan', 'missing')
    # dtx.loc[:,x_i] = np.where(pd.isnull(dtx[x_i]), dtx[x_i], dtx[x_i].astype(str))
    dtx = dtx.replace(np.nan, 'missing').assign(rowid = dtx.index).sort_values('rowid')
    #print(dtx)
    # rename binx
    binx.columns = ['bin', x_i, '_'.join([x_i,woe_points])]
    # merge
    dtx_suffix = pd.merge(dtx, binx, how='left', on=x_i).sort_values('rowid')      .set_index(dtx.index)[['_'.join([x_i,woe_points])]]
    return dtx_suffix

def scorecard(bins, model, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False):
    '''
    Creating a Scorecard
    ------
    `scorecard` creates a scorecard based on the results from `woebin` 
    and LogisticRegression of sklearn.linear_model
    Params
    ------
    bins: Binning information generated from `woebin` function.
    model: A LogisticRegression model object.
    points0: Target points, default 600.
    odds0: Target odds, default 1/19. Odds = p/(1-p).
    pdo: Points to Double the Odds, default 50.
    basepoints_eq0: Logical, default is FALSE. If it is TRUE, the 
      basepoints will equally distribute to each variable.
    
    Returns
    ------
    DataFrame
        scorecard dataframe
    
    Examples
    ------    
    # scorecard
    # Example I # creat a scorecard
    card = scorecard(bins, lr, X.columns)
    '''
    import re
    def ab(points0=600, odds0=1/19, pdo=50):
        b = pdo/np.log(2)
        a = points0 + b*np.log(odds0)
        return {'a':a, 'b':b}
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a = aabb['a']
    print(aabb)
    b = aabb['b']
    #变量系数名
    coef_names = model.params.index.tolist()[1:]
    #变量系数值
    coef_values = model.params.tolist()[1:]
    #截距项
    const = model.params[0] 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    xs = [re.sub('_woe$', '', i) for i in coef_names]
    # coefficients
    coef_df = pd.Series(coef_values, index=np.array(xs))      .loc[lambda x: x != 0]#.reset_index(drop=True)
    # scorecard
    len_x = len(coef_df)
    basepoints = a - b*const #评分公式展开后的A-B*截距项
    card = {}
    #basepoints是否平均到每一箱
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':0}, index=np.arange(1))
        
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]              .assign(points = lambda x: round(-b*x['woe']*coef_df[i] + basepoints/len_x))              [["variable", "bin", "points"]]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints)}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]              .assign(points = lambda x: round(-b*x['woe']*coef_df[i]))              [["variable", "bin", "points"]]
    return card

def scorecard_ply(dt, card, only_total_score=True, print_step=0, replace_blank_na=True, var_kp = None):
    dt = dt.copy(deep=True)
    # remove date/time col
    # dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    if replace_blank_na:
        blank_cols = [i for i in list(dt) if dt[i].astype(str).str.findall(r'^\s*$').apply(lambda x:0 if len(x)==0 else 1).sum()>0]
        if len(blank_cols) > 0:
            warnings.warn('There are blank strings in {} columns, which are replaced with NaN. \n (ColumnNames: {})'.format(len(blank_cols), ', '.join(blank_cols)))
            dt = dt.replace(r'^\s*$', np.nan, regex=True)
    # print_step 是否打印步骤
    if not isinstance(print_step, (int, float)) or print_step<0:
        warnings.warn("Incorrect inputs; print_step should be a non-negative integer. It was set to 1.")
        print_step = 1
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)
    elif isinstance(card, pd.DataFrame):
        card_df = card.copy(deep=True)
    # x variables
    xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
    #print(xs)
    # length of x variables
    xs_len = len(xs)
    # initial datasets
    dat = dt.loc[:,list(set(dt.columns)-set(xs))]
    
    # loop on x variables
    for i in np.arange(xs_len):
        x_i = xs[i] #获取每个x变量
        # print xs
        if print_step>0 and bool((i+1)%print_step): 
            print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i))
        
        cardx = card_df.loc[card_df['variable']==x_i] #某个变量的card
        dtx = dt[[x_i]] #某个变量结果
        # score transformation
        dtx_points = woepoints_ply1(dtx, cardx, x_i, woe_points="points")
        dat = pd.concat([dat, dtx_points], axis=1)
    # set basepoints 添加basepoint
    card_basepoints = list(card_df.loc[card_df['variable']=='basepoints','points'])[0] if 'basepoints' in card_df['variable'].unique() else 0
    # total score
    dat_score = dat[xs+'_points']
    dat_score.loc[:,'score'] = card_basepoints + dat_score.sum(axis=1)
    # dat_score = dat_score.assign(score = lambda x: card_basepoints + dat_score.sum(axis=1))
    # return
    if only_total_score: dat_score = dat_score[['score']]
    
    # check force kept variables
    if var_kp is not None:
        if isinstance(var_kp, str):
            var_kp = [var_kp]
        var_kp2 = list(set(var_kp) & set(list(dt)))
        len_diff_var_kp = len(var_kp) - len(var_kp2)
        if len_diff_var_kp > 0:
            warnings.warn("Incorrect inputs; there are {} var_kp variables are not exist in input data, which are removed from var_kp. \n {}".format(len_diff_var_kp, list(set(var_kp)-set(var_kp2))) )
        var_kp = var_kp2 if len(var_kp2)>0 else None
    if var_kp is not None: dat_score = pd.concat([dt[var_kp], dat_score], axis = 1)
    return dat_score

#载入相关包
import numpy as np
import pandas as pd
import scorecardpy as sc
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import sys
sys.path.append("/home/dfxb_w03040/.local/lib/python3.6/site-packages")
sys.path.append("/home/fxb_759995")
sys.path.append("/home/dfxb_w03040/jupyter_data/gaojianxin/Lxl_folder")

import statsmodels.api as sm
from LuckyDer import *


###############################################模型文档输出############################################
#模型文档之样本分布
'''
function:统计模型评分分布，及好坏分布
score:LR的评分,要求评分的列名为'score'
cuts:分组区间
data:带有好坏标签的数据，与score的编号/主题保持一致
'''
def score_cut_stat(score, data, cuts= range(200,1001,50), flag_name = 'flagy'):
    score['cut'] = pd.cut(score['score'],bins = cuts,right = False)
    score['cut'] = score['cut'].astype(str)
    score[flag_name] = data[flag_name]
    flgy1_sum = data[flag_name].agg(['sum']).values
    result = score.groupby("cut")[flag_name].agg(['count','sum']).assign(cn_pro = lambda x : np.round(x['count']/data.shape[0],2),
                                                                y_pro = lambda x : np.round(x['sum']/flgy1_sum,4))
    result = pd.DataFrame(result)
    f = lambda x : '%.2f%%' % (x*100)
    result[['y_pro']] = result[['y_pro']].applymap(f)
    order = ['count','cn_pro','sum','y_pro']
    result = result[order]
    return result 


#模型文档之IV值&缺失率
'''
data:需要计算IV和缺失率的数据
bins:该数据对应的分箱
'''
def Iv_MissRate(data,bins):
    woe  = sc.woebin_ply(data,bins = bins)
    iv_info = iv(woe,y = 'flagy')
    iv_info = iv_info.sort_values(by = 'info_value',ascending = False)
    iv_info['info_value'] = iv_info.info_value.round(4)
    iv_info['variable'] = [i.replace("_woe",'') for i in iv_info.variable.tolist()]
    ##计算缺失率
    tmp = miss(data[iv_info.variable])
    iv_info['miss_rate'] = tmp['miss_rate'].tolist()

    #数据输出
    df_iv = pd.DataFrame(columns = ['number','variable','exp','IV','miss_rate'])
    df_iv['number'] = list(range(1,iv_info.shape[0]+1))
    df_iv['variable'] = iv_info['variable'].tolist()
    df_iv['IV'] = iv_info['info_value'].tolist()
    df_iv['miss_rate'] = iv_info['miss_rate'].tolist()
    return df_iv

#模型文档之变量分箱
def var_bins(card,bins_train,bins_test):
    b = list()
    for i in card.keys():
        b.append(str(i))
    print(b)

    #输出各个变量在每一箱的分值
    bins_points = pd.DataFrame()
    for i in b[1:]:
        print(i)
        bi = pd.DataFrame(card[i])
        bins_points = bins_points.append(bi)

    #输出各个变量在每一箱的WOE值
    woe = pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_train[i]['woe'])
        woe = woe.append(bi)
    print("#################WOE值#############")
    print(woe)

    #输出各个变量的区间数目
    train_count =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_train[i]['count'])
        train_count =train_count.append(bi)
    print("#############区间样本量#############")
    print(train_count)

    #输出各个变量的区间占比
    train_count_distr =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_train[i]['count_distr'])
        train_count_distr =train_count_distr.append(bi)
    print("#############区间占比#############")
    print(train_count_distr)


    #输出各个变量的区间坏客户率
    train_badprob =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_train[i]['badprob'])
        train_badprob =train_badprob.append(bi)
    print("#########区间坏客户率#############")
    print(train_badprob)

    bins_points['woe'] = woe.values
    bins_points['train_count'] = train_count.values
    bins_points['train_count_distr'] = train_count_distr.values
    bins_points['train_badprob'] = train_badprob.values
    print(bins_points)


    #输出各个变量在每一箱的WOE值
    woe = pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_test[i]['woe'])
        woe = woe.append(bi)
    print("#################WOE值#############")
    print(woe)

    #输出各个变量的区间数目
    test_count =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_test[i]['count'])
        test_count =test_count.append(bi)
    print("#############区间样本量#############")
    print(test_count)

    #输出各个变量的区间占比
    test_count_distr =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_test[i]['count_distr'])
        test_count_distr =test_count_distr.append(bi)
    print("#############区间占比#############")
    print(test_count_distr)


    #输出各个变量的区间坏客户率
    test_badprob =  pd.DataFrame()
    for i in b[1:]:
        bi = pd.DataFrame(bins_test[i]['badprob'])
        test_badprob =test_badprob.append(bi)
    print("#########区间坏客户率#############")
    print(test_badprob)

    bins_points['test_count'] = test_count.values
    bins_points['test_count_distr'] = test_count_distr.values
    bins_points['test_badprob'] = test_badprob.values
    print(bins_points)

    ########直接用于模型文档的数据框
    bins_points_file = pd.DataFrame(columns = ['variable','exp','bin','train_count_distr','train_badprob','test_count_distr','test_badprob'])
    bins_points_file[['variable','bin','train_count_distr','train_badprob','test_count_distr','test_badprob']]= bins_points[['variable','bin','train_count_distr','train_badprob','test_count_distr','test_badprob']]
    return bins_points,bins_points_file


#模型文档之模型参数
def df_param(model,train_woe,fin_col,train,flag_name='flagy'):
    ####model.features
    model_feas = model.params.index.tolist()[0:]

    ####model.params####
    model_params = model.params.to_list()

    ####model.pvalues####
    model_pvalue = model.pvalues.tolist()

    ####model.stand_errors####
    model_std = model.bse.tolist()

    ####model.z_value####
    model_z=[a/b for a,b in zip(model_params,model_std)]

    ####vif########
    import statsmodels.stats.outliers_influence as oi
    import statsmodels.api as sm
    fin_col_woe = model.params.index.tolist()[1:]+ [flag_name]
    x_col = [i for i in fin_col_woe if i != flag_name]
    reg_x  =train_woe[x_col]

    # 看VIF值
    xs = np.array(sm.add_constant(reg_x), dtype=np.float)
    xs_name = ["const"] + reg_x.columns.tolist() # 需要 计算VIF的变量
    vif = pd.DataFrame([{"variable":xs_name[i], "vif":oi.variance_inflation_factor(xs, i)} for i in range(len(xs_name))])

    ####IV#######
    model_iv = iv(train_woe[fin_col_woe],y = flag_name)

    ####miss_rate#######
    x_col = [i for i in fin_col if i != flag_name]
    model_miss = miss(train[x_col])

    #合并数据
    df_param1 = pd.DataFrame(columns = ['Number','feas','exp','params','std','Z_value','P_value','VIF','missing'])
    df_param1['Number'] = list(range(0,len(model_feas)))
    df_param1['feas'] =  model_feas
    df_param1['params'] = model_params
    df_param1['std'] = model_std
    df_param1['Z_value'] = model_z
    df_param1['P_value'] = model_pvalue
    df_param1['VIF'] = vif['vif'].tolist()
    df_param1['missing'].iloc[1:len(model_feas)] = model_miss['miss_rate'].tolist()
    df_param =pd.merge(df_param1, model_iv, left_on = 'feas', right_on = 'variable', how = 'left' ).drop('variable',axis = 1)
    df_param['feas'] = [i.replace("_woe",'') for i in df_param.feas.tolist()]

    #调换顺序
    order = ['Number','feas','exp','params','std','Z_value','P_value','VIF','info_value','missing']
    df_param = df_param[order]
    f = lambda x:'%.4f'% x
    df_param[['params','std','Z_value','P_value','info_value']] = df_param[['params','std','Z_value','P_value','info_value']].applymap(f)
    return df_param

# #获取分箱的各个数值，并以列表的形式展示
# def break_list(bins):
#     d1 = {}
#     for  j in bins.keys():
#         x1 = [i for i in bins[j]['bin'].str.replace("missing|\[-inf\,|\)|,inf|\[","").tolist() if i  != '']
#         cut_list = []
#         for i in x1:
#             xx = i.split(",")
#             cut_list = cut_list + xx
#         cut_list = pd.Series([float(i) for i in set(cut_list)]).sort_values().tolist()
#         d1[j] = cut_list
#     return d1

#保存分箱图
'''
function：将多个变量的分箱图集中在一张画布上
bins_adj:分箱结果
model_list:分箱的特征名
save_addr:存储文件名
col_num:画布的列数
fisize:画布大小
'''
def save_png(bins_adj,model_list,save_addr,col_num = 1, figsize=(4,2)):
    path = os.getcwd()
    import matplotlib.pyplot as plt
    from matplotlib import image
    fig = plt.figure(figsize=figsize)             #设置画布大小
    img_list = []                               #img列表
    for i in model_list:
        sc.woebin_plot(bins_adj[i])
        plt.savefig("./png/"+i + '.png')      #存储每个图像 
        img = image.imread("./png/"+i+'.png') #数组方式读每个图像进去，需要新建个png文件夹
        img_list.append(img)                        #append
    img_list = np.array(img_list)                   #转换为numpy数组
    for i in range(0,len(img_list),1):         #获取img长度
        ax = plt.subplot(int(len(img_list)/1) +1,col_num,i+1)              #循环增加子图 n行4列
        ax.xaxis.set_major_locator(plt.NullLocator())  #去掉X轴标签
        ax.yaxis.set_major_locator(plt.NullLocator())  #去掉X轴标签
        plt.imshow(img_list[i])                        #画图
    plt.savefig(path+'//' + save_addr,dpi = 2000)                  #保存
    plt.close()                                 #关闭
