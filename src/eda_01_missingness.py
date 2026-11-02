# EDA step 1: missingness profile (run: python eda_01_missingness.py)
import pandas as pd
R=str(__import__("pathlib").Path(__file__).resolve().parents[1]/"data"/"raw")+"/"
m=pd.read_parquet(R+"municipalities.parquet"); a=pd.read_parquet(R+"audit_opinions.parquet"); u=pd.read_parquet(R+"uifwexp.parquet")
for n,d in [("muni",m),("audit",a),("uifw",u)]:
    print("==",n,d.shape); print(d.isna().sum().to_string()); print((d.astype(str).apply(lambda s:s.str.strip().isin(["","None","nan"]))).sum().to_string())
print(m.columns.tolist()); print(m['municipality.category'].value_counts())
print(a['financial_year_end.year'].value_counts().sort_index().to_string())
print(a['opinion.code'].value_counts())
print(u['financial_year_end.year'].value_counts().sort_index().to_string()); print(u['item.code'].value_counts())
print(a.duplicated(['demarcation.code','financial_year_end.year']).sum(), u.duplicated(['demarcation.code','financial_year_end.year','item.code']).sum())
print(u.amount.describe()); print((u.amount==0).sum(), (u.amount<0).sum())
import pandas as pd
R=str(__import__("pathlib").Path(__file__).resolve().parents[1]/"data"/"raw")+"/"
m=pd.read_parquet(R+"municipalities.parquet"); a=pd.read_parquet(R+"audit_opinions.parquet"); u=pd.read_parquet(R+"uifwexp.parquet")
K=['demarcation.code','financial_year_end.year']
u['miss']=u.amount.isna()
print(u.groupby('financial_year_end.year').miss.mean().round(3).to_string())
print(u.groupby('item.code').miss.mean().round(3))
# rows where all 3 items missing for a muni-year
w=u.pivot_table(index=K,columns='item.code',values='miss',aggfunc='first')
print("muni-years",len(w)," all3 missing",(w.sum(1)==3).sum()," some",((w.sum(1)>0)&(w.sum(1)<3)).sum())
x=u.merge(m[['municipality.demarcation_code','municipality.category','municipality.province_code']],left_on='demarcation.code',right_on='municipality.demarcation_code',how='left')
print("unmatched munis",x['municipality.category'].isna().sum(), x.loc[x['municipality.category'].isna(),'demarcation.code'].unique()[:20])
print(x.groupby('municipality.category').miss.mean().round(3)); print(x.groupby('municipality.province_code').miss.mean().round(3).to_string())
y=x.merge(a[K+['opinion.code']],on=K,how='left')
print(y.groupby('opinion.code',dropna=False).miss.agg(['mean','size']).round(3))
# audit panel completeness: munis x years
print("audit munis",a['demarcation.code'].nunique(), "uifw munis",u['demarcation.code'].nunique(), "muni table",len(m))
cur=set(a.loc[a['financial_year_end.year']==2024,'demarcation.code'])
p=a[a['demarcation.code'].isin(cur)].groupby('demarcation.code').size(); print("current munis years covered",p.value_counts().sort_index().to_string())
print("audit keys missing in uifw", len(a.merge(u[K].drop_duplicates(),on=K,how='left',indicator=True).query('_merge=="left_only"')))
print(a[a['opinion.code']=='outstanding'].groupby('financial_year_end.year').size().to_string())
print(a[a['opinion.report_url'].isna()].groupby('financial_year_end.year').size().to_string())
print(u[u.amount<0])
import pandas as pd
R=str(__import__("pathlib").Path(__file__).resolve().parents[1]/"data"/"raw")+"/"
a=pd.read_parquet(R+"audit_opinions.parquet")
K=['demarcation.code','financial_year_end.year']
panel=a[a['financial_year_end.year'].between(2018,2024)][K+['opinion.code']]
print("panel muni-years 2018-24:",len(panel))
for c in ["incexp_v2","financial_position_v2","aged_debtor_v2"]:
    d=pd.read_parquet(R+c+".parquet"); print("==",c,d.shape, d.columns.tolist())
    print(" nulls:",d.isna().sum()[d.isna().sum()>0].to_dict())
    print(" amount types:",d['amount_type.code'].value_counts().to_dict())
    au=d[d['amount_type.code']=='AUDA']
    print(" AUDA years:",au['financial_year_end.year'].value_counts().sort_index().to_dict(), "items",au['item.code'].nunique())
    have=au[K].drop_duplicates(); have['has']=1
    p=panel.merge(have,on=K,how='left'); p['has']=p.has.fillna(0)
    print(" panel muni-years with NO audited figures:",int((p.has==0).sum()), f"({(p.has==0).mean():.1%})")
    print("  by year:",p.groupby('financial_year_end.year').has.apply(lambda s:int((s==0).sum())).to_dict())
    print("  missing rate by opinion:",p.groupby('opinion.code').has.apply(lambda s:round((s==0).mean(),3)).to_dict())
    # item completeness among munis that report
    cnt=au.groupby(K)['item.code'].nunique(); print("  items per muni-year: median",cnt.median(),"min",cnt.min(),"max",cnt.max())
    v=[x for x in d.columns if x.endswith('.sum')]
    print("  zeros share:",{x:round((au[x]==0).mean(),3) for x in v[:3]})
