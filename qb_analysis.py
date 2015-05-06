
# coding: utf-8

# In[685]:

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib2
import string
from pylab import *
get_ipython().magic(u'matplotlib inline')


# In[686]:

filepath='/Users/DanLo1108/Documents/Projects/NFL Draft/'


# In[687]:

#Loop through years of combine stats to extract urls
urls=[]
years=np.arange(1999,2016)
for year in years:
    url='http://nflcombineresults.com/nflcombinedata.php?year='+str(year)+'&pos=QB&college='
    urls.append(url)


# In[691]:

#Initializes stat lists
years=[]
names=[]
colleges=[]
height=[]
weight=[]
wonderlic=[]
forty=[]
bench=[]
vert=[]
broad=[]
shuttle=[]
cone=[]


# In[692]:

#Loop through years of combine data
for url in urls:
    
    #Gets content of url webpage
    request=urllib2.Request(url)
    page = urllib2.urlopen(request)
    
    #Reads content of page
    content=page.read()
    soup=BeautifulSoup(content,'lxml')

    #Gets results from table
    table=soup.find_all('table')
    results=table[0].find_all('td')
    
    #Loops through results and appends item to appropriate list
    count = 0
    for item in results[13:]:
        count += 1
        if np.mod(count,13) == 1 and item.string != u'\xa0':
            years.append(item.string)
        if np.mod(count,13) == 2:
            names.append(item.string)
        if np.mod(count,13) == 3:
            colleges.append(item.string)
        if np.mod(count,13) == 5:
            height.append(item.string)
        if np.mod(count,13) == 6:
            weight.append(item.string)
        if np.mod(count,13) == 7:
            wonderlic.append(item.string)
        if np.mod(count,13) == 8:
            forty.append(item.string)
        if np.mod(count,13) == 9:
            bench.append(item.string)
        if np.mod(count,13) == 10:
            vert.append(item.string)
        if np.mod(count,13) == 11:
            broad.append(item.string)
        if np.mod(count,13) == 12:
            shuttle.append(item.string)
        if np.mod(count,13) == 0:
            cone.append(item.string)


# In[693]:

#Create combine stats dataframe
combine_stats=pd.DataFrame({'Year':years,
                            'Name':names,
                            'College':colleges,
                            'Height':height,
                            'Weight':weight,
                            'Wonderlic':wonderlic,
                            'Forty':forty,
                            'Bench':bench,
                            'Vert':vert,
                            'Broad':broad,
                            'Shuttle':shuttle,
                            'Cone':cone})


combine_stats=combine_stats[['Year','Name','College','Height','Weight','Wonderlic',
                             'Forty','Bench','Vert','Broad','Shuttle','Cone']]


# In[445]:

#Strip punctuation from player names (helps with joining later)
import re
combine_stats['Name']=map(lambda x: re.sub(r"\.'",'',x),np.array(combine_stats.Name))


# In[696]:

#Function to get stats from web page results
def get_stats(results,stats_dict):

    count1=0
    for result in results[0]:
        count1+=1
        if count1==2:
            count2=0
            for res in result:
                count2+=1
                if count2 == 16:
                    stats_dict['Pass_Att'].append(float(res.string))
                if count2 == 18:
                    stats_dict['Comp_perc'].append(float(res.string))
                if count2 == 20:
                    stats_dict['Pass_Yards'].append(float(res.string))
                if count2 == 22:
                    stats_dict['Pass_Yards_Att'].append(float(res.string))
                if count2 == 26:
                    stats_dict['Pass_TD'].append(float(res.string))
                if count2 == 28:
                    stats_dict['Int'].append(float(res.string))
                if count2 == 30:
                    stats_dict['Rating'].append(float(res.string))

    count1=0
    for result in results[1]:
        count1+=1
        if count1==2:
            count2=0
            for res in result:
                count2+=1
                if count2 == 14:
                    stats_dict['Rush_Att'].append(float(res.string))
                if count2 == 16:
                    stats_dict['Rush_Yards'].append(float(res.string))
                if count2 == 20:
                    stats_dict['Rush_TD'].append(float(res.string))


# In[698]:

#Get a list of all drafted QBs

url='http://www.nfl.com/draft/history/fulldraft?position=QB&type=position'

#Gets content of url webpage
request=urllib2.Request(url)
page = urllib2.urlopen(request)

#Reads content of page
content=page.read()
soup=BeautifulSoup(content,'lxml')

results=soup.find_all('a')
drafted_qbs=[]
for r in results:
    player = r.string
    if r.string in np.array(combine_stats.Name):
        drafted_qbs.append(str(r.string))


# In[448]:

#Manually add QBs whose names have periods (there is inconsistency in data source)
drafted_qbs.append('JP Losman')
drafted_qbs.append('BJ Symons')
drafted_qbs.append('JT OSullivan')
drafted_qbs.append('AJ Feeley')
drafted_qbs.append('BJ Daniels')
drafted_qbs.append('EJ Manuel')
drafted_qbs.append('AJ McCarron')
drafted_qbs.append('TJ Yates')
#Manually add 2015 draftees
drafted_qbs.append('Jameis Winston')
drafted_qbs.append('Marcus Mariota')
drafted_qbs.append('Garrett Grayson')
drafted_qbs.append('Sean Mannion')
drafted_qbs.append('Bryce Petty')
drafted_qbs.append('Brett Hundley')


# In[449]:

#Initialize stats dictionary
stats_dict={'Name':[],
            'Pass_Att':[],
            'Comp_perc':[],
            'Pass_Yards':[],
            'Pass_Yards_Att':[],
            'Pass_TD':[],
            'Int':[],
            'Rating':[],
            'Rush_Att':[],
            'Rush_Yards':[],
            'Rush_TD':[]}


# In[450]:

#Loop through player names
import string
names=drafted_qbs
bad_names=[]
for name in names:
    
    try:
        #Get player url
        name_lower=string.lower(name)
        url_name=string.split(name_lower)[0]+'-'+string.split(name_lower)[1]+'-1'
        url='http://www.sports-reference.com/cfb/players/'+url_name+'.html'

        #Gets content of url webpage
        request=urllib2.Request(url)
        page = urllib2.urlopen(request)

        #Reads content of page
        content=page.read()
        soup=BeautifulSoup(content,'lxml')

        #Gets career statistics in footings
        results=soup.find_all('tfoot')
        
        get_stats(results,stats_dict)
        stats_dict['Name'].append(name)
        
    except:
        bad_names.append(name)


# In[451]:

#We can see that the lengths of these stat lists are not equal
for stat in stats_dict:
    print len(stats_dict[stat])


# In[452]:

#Remove "bad names" from names
names=[n for n in names if n not in bad_names]


# In[453]:

#Reinitialize stats dictionary
stats_dict={'Name':[],
            'Pass_Att':[],
            'Comp_perc':[],
            'Pass_Yards':[],
            'Pass_Yards_Att':[],
            'Pass_TD':[],
            'Int':[],
            'Rating':[],
            'Rush_Att':[],
            'Rush_Yards':[],
            'Rush_TD':[]}


# In[455]:

#Loop through player names
for name in names:
    
    try:
        #Get player url
        name_lower=string.lower(name)
        url_name=string.split(name_lower)[0]+'-'+string.split(name_lower)[1]+'-1'
        url='http://www.sports-reference.com/cfb/players/'+url_name+'.html'

        #Gets content of url webpage
        request=urllib2.Request(url)
        page = urllib2.urlopen(request)

        #Reads content of page
        content=page.read()
        soup=BeautifulSoup(content,'lxml')

        #Gets career statistics in footings
        results=soup.find_all('tfoot')
        
        get_stats(results,stats_dict)
        stats_dict['Name'].append(name)
        
    except:
        bad_names.append(name)


# In[456]:

#Verify all the same length
for stat in stats_dict:
    print len(stats_dict[stat])


# In[457]:

college_stats=pd.DataFrame()
for stat in stats_dict:
    college_stats[stat]=stats_dict[stat]


# In[458]:

bad_names


# In[459]:

#manually find stats for players whose names didn't register in the above web scraping.
#Note: a few players (J.T. O'Sullivan, Spergon Wynn, Chris Daft...) are removed from my
#database because I could not find their stats online (small school > 10 yrs ago)

bad_name_stats={'Name':['Jimmy Garoppolo','Brad Sorensen','John Skelton','Joe Flacco',"Kevin O'Connell",
                        'John David Booty','Josh Johnson','Ingle Martin','Alex Smith','Ryan Fitzpatrick',
                        'Brian St. Pierre'],
           'Pass_Att':[1668,1254,1363,942,1151,832,1065,87,587,454,803],
           'Comp_perc':[62.8,65.6,58.8,60.7,57.7,62.3,68.0,62.1,66.3,58.4,56.9],
           'Pass_Yards':[13156,9445,9923,7057,7689,6125,9699,750,5203,3756,5837],
           'Pass_Yards_Att':[7.9,7.5,7.3,7.2,6.7,7.4,9.1,8.6,8.9,8.3,7.3],
           'Pass_TD':[118,61,69,41,46,55,113,3,47,29,48],
           'Int':[51,27,36,15,34,21,15,2,8,14,32],
           'Rating':[146.3,135.0,131.4,131.7,121.1,140.9,176.7,141.3,164.4,142.8,129.7],
           'Rush_Att':[260,202,296,153,395,61,307,25,286,227,172],
           'Rush_Yards':[-137,-117,565,76,1312,-180,1864,70,1072,878,419],
           'Rush_TD':[8,6,14,9,19,2,19,0,15,11,2]
           }

new_college_stats=pd.DataFrame()
for stat in bad_name_stats:
    new_college_stats[stat]=bad_name_stats[stat]


# In[460]:

#Append bad names stats to rest of college stats
college_stats=college_stats.append(new_college_stats).reset_index().drop('index',axis=1)


# In[470]:

college_stats[college_stats.Name=='Tarvaris Jackson']


# In[462]:

#Fix RGIII stats
college_stats.ix[18,'Pass_Att']=1192
college_stats.ix[18,'Comp_perc']=67.1
college_stats.ix[18,'Pass_Yards']=10366
college_stats.ix[18,'Pass_Yards_Att']=8.7
college_stats.ix[18,'Pass_TD']=78
college_stats.ix[18,'Int']=17
college_stats.ix[18,'Rating']=158.9
college_stats.ix[18,'Rush_Att']=528
college_stats.ix[18,'Rush_Yards']=2254
college_stats.ix[18,'Rush_TD']=33

college_stats[college_stats.Name=='Robert Griffin']


#Fix Tarvaris Jackson stats (most were compiled at Alabama St)
college_stats.ix[76,'Pass_Att']=1022
college_stats.ix[76,'Comp_perc']=53.8
college_stats.ix[76,'Pass_Yards']=7964
college_stats.ix[76,'Pass_Yards_Att']=7.8
college_stats.ix[76,'Pass_TD']=68
college_stats.ix[76,'Int']=29
college_stats.ix[76,'Rating']=135.4
college_stats.ix[76,'Rush_Att']=275
college_stats.ix[76,'Rush_Yards']=1029
college_stats.ix[76,'Rush_TD']=11

#Fix BJ Coleman stats (most were compiled at Chattanooga)
college_stats.ix[26,'Pass_Att']=1016
college_stats.ix[26,'Comp_perc']=57.4
college_stats.ix[26,'Pass_Yards']=6892
college_stats.ix[26,'Pass_Yards_Att']=6.8
college_stats.ix[26,'Pass_TD']=52
college_stats.ix[26,'Int']=32
college_stats.ix[26,'Rating']=125.0
college_stats.ix[26,'Rush_Att']=103
college_stats.ix[26,'Rush_Yards']=-132
college_stats.ix[26,'Rush_TD']=7


#There may be a few others that should be adjusted but these stood out as being obvious outliers


# In[464]:

#Convert stats to rates by attempt
college_stats['Pass_TD_Rate']=college_stats.Pass_TD/college_stats.Pass_Att
college_stats['Int_Rate']=college_stats.Int/college_stats.Pass_Att
college_stats['Rush_Yards_Att']=college_stats.Rush_Yards/college_stats.Rush_Att
college_stats['Rush_TD_Rate']=college_stats.Rush_TD/college_stats.Rush_Att

#Pass/Rush ratio
college_stats['Pass_Rush_Rate']=college_stats.Pass_Att/college_stats.Rush_Att

college_stats=college_stats[['Name','Pass_Att','Comp_perc','Pass_Yards_Att','Pass_TD_Rate','Int_Rate','Rating','Rush_Att',
                             'Rush_Yards_Att','Rush_TD_Rate','Pass_Rush_Rate']]


# In[465]:

#Merge college and combine stats
qb_stats=college_stats.merge(combine_stats,'left')


# In[476]:

filepath='/Users/DanLo1108/Documents/Projects/NFL Draft/'
qb_stats.to_csv(filepath+'temp_data.csv',index=False)


# In[488]:

count=0
for i in qb_stats.index.values:
    try:
        a = float(qb_stats.ix[i].Forty)
        count+=1
    except:
        continue
print count


# In[136]:

#New variable; "2" for BCS school, "1" for non-BCS FBS, "0" for FCS

BCS=[2,2,2,1,2,2,2,1,1,2,
     2,2,2,2,2,2,1,2,2,2,
     2,2,2,2,2,1,1,1,2,2,
     2,2,1,1,2,2,1,2,2,2,
     2,2,2,2,1,2,1,2,2,2,
     2,2,2,2,1,1,1,2,2,2,
     2,2,2,2,2,2,1,1,2,2,
     1,1,2,2,2,2,1,2,2,1,
     1,2,2,2,1,2,2,2,2,2,
     2,2,2,1,1,2,1,2,1,2,
     2,2,1,2,1,2,2,2,2,2,
     2,1,2,1,1,1,2,2,2,1,
     2,2,2,2,2,2,2,2,2,2,
     1,1,2,2,2,2,1,2,2,2,
     2,2,1,2,1,2,2,2,2,2,
     2,2,1,2,2,2,0,0,0,0,
     1,2,0,0,1,0,2,2,2,2]

qb_stats['BCS']=np.array(BCS)


# In[599]:

#Update Wonderlic scores for last 2 years

qb_stats.ix[4,'Wonderlic']=29
qb_stats.ix[5,'Wonderlic']=29
qb_stats.ix[6,'Wonderlic']=30
qb_stats.ix[7,'Wonderlic']=25
qb_stats.ix[8,'Wonderlic']=29
qb_stats.ix[9,'Wonderlic']=25

qb_stats.ix[154,'Wonderlic']=27
qb_stats.ix[155,'Wonderlic']=33
qb_stats.ix[156,'Wonderlic']=20
qb_stats.ix[157,'Wonderlic']=40
qb_stats.ix[158,'Wonderlic']=31
qb_stats.ix[159,'Wonderlic']=26


# In[635]:

#I noticed that EJ Manuel, AJ McCarron and TJ Yates aren't on the list.
#I believe there are a few others but no one notable (had trouble matching on acronym names).
#I'll add these three and assume that any others that may have been 
#omitted won't significantly affect this study

omitted_qbs=pd.DataFrame({'Name':['E.J. Manuel','A.J. McCarron','T.J. Yates'],
                          'College':['Florida State','Alabama','North Carolina'],
                          'Height':[77,75,75],
                          'Weight':[237,220,219],
                          'Wonderlic':[28,22,25],
                          'Forty':[4.65,4.94,5.06],
                          'Bench':[None,None,None],
                          'Vert':[34.0,28.0,29.5],
                          'Broad':[118,99,104],
                          'Shuttle':[4.21,None,4.12],
                          'Cone':[7.08,None,6.96],
                          'Pass_Att':[897,1026,1277],
                          'Comp_perc':[66.9,66.9,62.3],
                          'Pass_Yards_Att':[8.6,8.8,7.3],
                          'Pass_TD_Rate':[.052,.075,.045],
                          'Int_Rate':[.031,.015,.036],
                          'Rating':[150.4,162.5,131.7],
                          'Rush_Att':[298,119,220],
                          'Rush_Yards_Att':[2.8,-.4,-1.5],
                          'Rush_TD_Rate':[.037,.025,.032],
                          'Pass_Rush_Rate':[3.01,8.62,5.80],
                          'BCS':[1,1,1],
                          'Year':[2013,2014,2011]})

qb_stats=qb_stats.append(omitted_qbs).reset_index().drop('index',axis=1)


# In[139]:

qb_stats.to_csv(filepath+'temp_2.csv',index=False)


# In[4]:

qb_stats=pd.read_csv(filepath+'temp_2.csv')


# In[716]:

#See how many missing values are in each combine stat:

#Wonderlic: 20/170 missing
#Forty: 30/170 missing
#Bench: 165/170
#Broad: 33/170
#Shuttle: 46/170
#Cone: 47/170
#Vert: 30/170


count=0
for b in qb_stats.Vert:
    try:
        a=1+float(b)
    except:
        count+=1
        
print count


# In[696]:

#strip * from forty times:
qb_stats['Forty']=map(lambda x: re.sub('\*','',str(x)),qb_stats.Forty)


# In[826]:

#Convert to numeric values:
def conv_numeric(x,stat):
    try:
        return float(x[stat])
    except:
        return np.nan
    
    
qb_stats['Height']=qb_stats.apply(lambda x: conv_numeric(x,'Height'),axis=1)
qb_stats['Weight']=qb_stats.apply(lambda x: conv_numeric(x,'Weight'),axis=1)
qb_stats['Wonderlic']=qb_stats.apply(lambda x: conv_numeric(x,'Wonderlic'),axis=1)
qb_stats['Bench']=qb_stats.apply(lambda x: conv_numeric(x,'Bench'),axis=1)
qb_stats['Forty']=qb_stats.apply(lambda x: conv_numeric(x,'Forty'),axis=1)
qb_stats['Vert']=qb_stats.apply(lambda x: conv_numeric(x,'Vert'),axis=1)
qb_stats['Broad']=qb_stats.apply(lambda x: conv_numeric(x,'Broad'),axis=1)
qb_stats['Shuttle']=qb_stats.apply(lambda x: conv_numeric(x,'Shuttle'),axis=1)
qb_stats['Cone']=qb_stats.apply(lambda x: conv_numeric(x,'Cone'),axis=1)


# In[10]:

#For missing wonderlic values, give score of 25
for i in qb_stats.index.values:
    if np.isnan(qb_stats.ix[i].Wonderlic):
        qb_stats.ix[i,'Wonderlic']=25


# In[862]:

stats=qb_stats.drop(['Year','College','Bench','Vert','Broad','Shuttle','Cone'],axis=1)


# In[277]:

stats


# In[280]:

def normalize(x,data,stat):
    return (x[stat]-np.mean(data[stat]))/np.std(data[stat])

for stat in stats:
    if stat != 'Name':
        stats[stat]=stats.apply(lambda x: normalize(x,stats,stat),axis=1)


# In[1008]:

#Use data from factor analysis, weigh by each factor's explained variance
stats=fa_stats


# In[1016]:

#Find most similar players using euclidean distance
from sklearn.neighbors import DistanceMetric

names=[]
scores=[]
p_name='Brett Hundley'
p1=np.array(stats[stats.Name==p_name].drop('Name',axis=1))[0]
for name in sorted(stats.Name):
    dist=DistanceMetric.get_metric('euclidean')
    p2=np.array(stats[stats.Name==name].drop('Name',axis=1))[0]
    distance=dist.pairwise([p1,p2])[1,0]
    if name != p_name:
        scores.append(distance)
        names.append(name)
        

similarities=pd.DataFrame({'Player':np.array(names),
                           'Similarity':np.array(scores)})


similarities.sort('Similarity')


# In[868]:

#Run a PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

X=stats.drop('Name',axis=1)

pca=PCA()
pca.fit(X)


# In[869]:

#Get explained variance
ev=pca.explained_variance_ratio_
var=[]
cum_var=[]
for i in range(len(ev)):
    var.append(ev[i])
    cum_var.append(sum(var))


# In[147]:

cum_var


# In[873]:

#Plot scree plot
plot(np.arange(1,16),var)
xlabel('Component',fontsize=16)
ylabel('Percent of variance explained',fontsize=16)
title('PCA Scree plot',fontsize=20)


# In[154]:

#6 components
pca=PCA(n_components=6)
X_comp=pca.fit_transform(X)


# In[155]:

components=pd.DataFrame({'Variable':np.array(X.columns)})
for i in range(len(pca.components_)):
    components['comp'+str(i)]=pca.components_[i]


# In[156]:

components


# In[157]:

pc_stats=pd.DataFrame({'Name':np.array(stats.Name)})
for i in range(len(X_comp.T)):
    pc_stats['comp'+str(i)]=X_comp.T[i]


# In[158]:

pc_stats


# In[310]:

#4 Factors explain 65% of variance are are easily interpretable
fa=FactorAnalysis(n_components=4)
fa.fit(X)


# In[311]:

factors=pd.DataFrame({'Variable':np.array(X.columns)})
for i in range(len(fa.components_)):
    factors['fact'+str(i+1)]=fa.components_[i]


# In[312]:

factors


# In[1013]:

X_fa=fa.fit_transform(X)
fa_stats=pd.DataFrame({'Name':np.array(stats.Name)})
for i in range(len(X_fa.T)):
    fa_stats['fact'+str(i+1)]=X_fa.T[i]


# In[997]:

#Run K-Means Clustering on Factors
from sklearn.cluster import KMeans

num_clusters=5

clust=KMeans(n_clusters=num_clusters)
clust.fit(fa_stats.drop('Name',axis=1))


# In[998]:

clust.cluster_centers_.T


# In[999]:

centroids=clust.cluster_centers_.T[:][0:2]


# In[1000]:

centroids


# In[1001]:

fa_stats['Cluster']=clust.predict(fa_stats.drop('Name',axis=1))


# In[1002]:

for i in range(num_clusters):
    plot(fa_stats[fa_stats.Cluster==i]['fact1'],fa_stats[fa_stats.Cluster==i]['fact2'],'.',label='Cluster '+str(i))
    legend(loc='center left', bbox_to_anchor=(1, .826))
    plot(centroids[0][i],centroids[1][i],marker='o',color='black')


# In[1003]:

#Add round where they were drafted:

draft_round=[1,1,1,2,4,4,6,6,6,7,
             2,3,4,4,4,4,7,1,1,1,
             1,2,3,3,4,6,7,7,1,1,
             1,1,2,2,3,5,5,6,7,1,
             1,2,3,4,6,6,7,7,7,1,
             1,1,2,4,5,5,6,6,1,2,
             2,5,7,7,1,1,2,2,2,3,
             5,6,1,1,1,2,2,3,3,5,
             6,7,1,1,3,3,4,4,5,5,
             6,1,1,1,1,3,4,4,6,6,
             7,7,7,1,1,1,1,3,3,4,
             6,1,1,1,3,4,4,5,5,5,
             7,1,2,2,2,4,4,5,6,6,
             1,3,5,6,6,6,7,7,7,1,
             1,1,1,1,2,3,4,4,7,7,
             1,1,3,3,4,5,2,7,5,1,
             3,4,5,5,1,7,5,1,5,5]

fa_stats['Round'] = np.array(draft_round)


# In[1004]:

#Preprocess for D3 visualization:

#Scatterplot radius based on draft round:
def get_size(x):
    return 1/sqrt(x.Round*.8)

fa_stats['size']=fa_stats.apply(lambda x: get_size(x), axis=1)


#Convert cluster name to string
def conv_cluster(x,data):
    for clust in fa_stats.Cluster.unique():
        if x.Cluster == clust:
            return 'Group '+str(clust+1)
    
    
fa_stats['Cluster']=fa_stats.apply(lambda x: conv_cluster(x,fa_stats), axis=1)


#Change column names:
fa_stats.columns=['Name','Pass_Efficiency_Score','Rush_Efficiency_Score','Experience_Score','Measurables_Score',
                  'Cluster','Round','size']


# In[1005]:

fa_stats.sort('Cluster').to_csv(filepath+'FA_stats.csv',index=False)


# In[1006]:

stats.columns.tolist()


# In[ ]:



