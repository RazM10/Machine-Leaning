
# coding: utf-8

# In[34]:


import pandas as pd
import MICE  # For MICE package in python is import
import iMICE  # For iMICE package in python is import
from fancyimpute import KNN  # For fancyimpute package in python is import
import kmeans # For kmeans package in python is import
import numpy as np  # All functions will be loaded into the local namespace.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score  # For making accuracy table
from sklearn.metrics import precision_score  # For making precision table
from sklearn.metrics import recall_score  # For making recall table
from sklearn.metrics import f1_score  # For making f1_score table

if __name__ == '__main__':
        

    # ### Input dataset

    # In[35]:


    df = pd.read_csv('age2001.csv')  # 


    # ### Delete all missing values from the original dataset

    # In[36]:


    df = df.dropna()  # df -> DataFrame; Use for remove missing values, if any rows or columns have.
    df = df[['Patient Name', 'Gender', 'AGE', 'Department', 'Sample', 'Address']]
    df.head()


    # ### Randomly select ()% from the original dataset

    # In[37]:


    df = df.sample(int(0.1 * len(df)))


    # ### Map gender to numeric

    # In[38]:


    gendermap = {"Male": 0, "Female": 1}
    df.replace({"Gender": gendermap}, inplace = True)


    # ### Map department to numeric

    # In[39]:


    deptmap = {'HAEMATOLOGY': 1, 'CLINICAL PATHOLOGY' : 2, 'BIOCHEMISTRY' : 3,
           'MICROBIOLOGY': 4, 'SEROLOGY': 5, 'IMMUNOLOGY': 6, 'PCR LAB': 7,
           'CYTO-PATHOLOGY': 8, 'HISTO-PATHOLOGY': 9, 'DELIVERY' : 10, 'VACCINATION CENTER': 11}
    df.replace({"Department": deptmap}, inplace = True)


    # ### Map sample to numeric

    # In[40]:


    sample_map = {'BLOOD' : 1, 'URINE' : 2, 'BLOOD.' : 3, 'BLOOD PLASMA' : 4, 'STOOL' : 5, 'SPUTUM' : 6,
           'PROSTATIC SMEAR' : 7, 'As Required' : 8, 'THROAT SWAB' : 9, 'URINE(SPOT)' : 10,
           'PUS' : 11, 'SEMEN' : 12, 'HVS' : 13, 'EAR (AURAL) SWAB' : 14, 'NAIL SCRAPING' : 15,
           'SPUTUM.' : 16, 'ET Tube' : 17, 'FLUID' : 18, 'FLUID.' : 19, 'PLEURAL FLUID' : 20,
           'CERVICAL SWAB' : 21, 'SKIN SCRAPING' : 22, 'SERUM' : 23, 'BLOOD & URINE' : 24,
           'UMBITICAL SWAB' : 25, 'URETHRAL DISCHARGE' : 26, 'WOUND SWAB' : 27,
           'CONJUNCTIVAL SWAB' :28, 'SYNOVIAL FLUID' : 29, 'IT Tube Tip' : 30}
    df.replace({"Sample": sample_map}, inplace = True)
    df = df.reset_index(drop = True)


    # ### Map "AGE" to "Binary Age Code"

    # In[41]:


    df['Binary Age Code'] = df['AGE'].apply(lambda x: 1 if x >= 19 else 0)


    # ### Apply NameValue Algorithm

    # In[42]:


    salutation = ["a v m", "admiraal", "air cdre", "air commodore", "air marshal", "air vice marshal", "alderman", "alhaji", "ambassador", "baron", "barones", "brig", "brig gen", "brig general", "brigadier", "brigadier general", "brother", "canon", "capt", "captain", "cardinal", "cdr", "chief", "cik", "cmdr", "col", "col dr", "colonel", "commandant", "commander", "commissioner", "commodore", "comte", "comtessa", "congressman", "conseiller", "consul", "conte", "contessa", "corporal", "councillor", "count", "countess", "crown prince", "crown princess", "dame", "datin", "dato", "datuk", "datuk seri",
            "deacon", "deaconess", "dean", "dhr", "dipl ing", "doctor", "dott", "dott sa", "dr", "dr ing", "dra", "drs", "embajador", "embajadora", "en", "encik", "eng", "eur ing", "exma sra", "exmo sr", "f o", "father", "first lieutient", "first officer", "flt lieut", "flying officer", "fr", "frau", "fraulein", "fru", "gen", "generaal", "general", "governor", "graaf", "gravin", "group captain", "grp capt", "h e dr", "h h", "h m", "h r h", "hajah", "haji", "hajim", "her highness", "her majesty", "herr", "high chief", "his highness",
            "his holiness", "his majesty", "hon", "hr", "hra", "ing", "ir", "jonkheer", "judge", "justice", "khun ying", "kolonel", "lady", "lcda", "lic", "lieut", "lieut cdr", "lieut col", "lieut gen", "lord", "m", "m l", "m r", "madame", "mademoiselle", "maj gen", "major", "master", "mevrouw", "miss", "mlle", "mme", "monsieur", "monsignor", "mr", "mrs", "ms", "mstr", "nti", "pastor", "president", "prince", "princess", "princesse", "prinses", "prof", "prof dr", "prof sir", "professor",
            "puan", "puan sri", "rabbi", "rear admiral", "rev", "rev canon", "rev dr", "rev mother", "reverend", "rva", "senator", "sergeant", "sheikh", "sheikha", "sig", "sig na", "sig ra", "sir", "sister", "sqn ldr", "sr", "sr d", "sra", "srta", "sultan", "tan sri", "tan sri dato", "tengku", "teuku",
            "mrs.", "mr.", "most.", "md.", "master.", "ms.", "alhaz", "hazi", "haji", "mohammad", "muhammad", "dr.", "miss", "prof.", "kazi", "engineer"] 

    def processName(name):
        name = name.lower()  # Make name all letter in lowercase 
        namesplits = name.split()  # To break a large string down into smaller
        for sals in salutation:
            if sals in namesplits:
                namesplits.remove(sals) # To remove name salutation and save in namesplits variable
        
        trans = str.maketrans('', '', 'aeiou')  # The method maketrans() returns a translation table that maps each character in the intabstring into the character at the same position in the outtab string. outtab − This is the string having corresponding mapping character.
        for i, nams in enumerate(namesplits):  # enumerate() - It returns an enumerate object.
            namesplits[i] = nams[0] + nams[1:].translate(trans)  # translate() method takes the translation table to replace/translate characters in the given string as per the mapping table.
        
        name = ' '.join(namesplits)  # By join() - simply concatenate two strings
        
        name = name.replace('g', 'j')
        name = name.replace('z', 'j')
        name = name.replace('q', 'k')
        
        trans = str.maketrans('aeioubcdfhjklmnprstvwxy .-', 'tuvwxabcdefghijklmnopqrsyz')  # The string maketrans() method returns a mapping table for translation usable for translate() method.
        name = name.translate(trans)  # The string translate() method returns a string where each character is mapped to its corresponding character in the translation table.
        
        return name


    # In[43]:


    df['NameValue'] = df['Patient Name'].map(processName)  


    # ### Apply geocode format

    # In[44]:


    fileLoc = open('demoLocations.txt', 'r')
    fileCode = open('demoGeoCodes.txt', 'r')
    dataLoc = fileLoc.readlines()
    dataCode = fileCode.readlines()
    codes = {}
    for i, v in enumerate(dataLoc):
        codes[v.strip()] = dataCode[i].strip()
    df['GeoCode'] = df['Address'].map(lambda x: codes[x])  # To make all the address as correspondence in GeoCode


    # ### AGE generalization by range

    # In[45]:


    def processAge(x):
        if x >= 0 and x <= 9:
            return 0
        elif x >= 10 and x <= 19:
            return 1
        elif x >= 20 and x <= 29:
            return 2
        elif x >= 30 and x <= 39:
            return 3
        elif x >= 40 and x <= 49:
            return 4
        elif x >= 50 and x <= 59:
            return 5
        elif x >= 60 and x <= 69:
            return 6
        elif x >= 70 and x <= 79:
            return 7
        elif x >= 80 and x <= 89:
            return 8
        elif x >= 90 and x <= 99:
            return 9


    # ### Map AGE to Age Code

    # In[46]:


    df['Age Code'] = df['AGE'].map(processAge)


    # ### Privacy preserved and generilized data

    # In[47]:


    t = df[['NameValue', 'Gender', 'AGE', 'Binary Age Code', 'Age Code', 'Department', 'Sample', 'GeoCode']]
    t.to_csv('0.Generilized.csv', index=False)  # It's print a ".csv" file with Privacy preserved and generilized data
    t.head(10)  


    # In[48]:


    actual = df['Gender'] # changable   # Save real DataFrame in a variable


    # ### Randomly inject ()% missing data with file print

    # In[49]:


    update = df.sample(int(0.05 * len(df))).index  #0.1 = 10%
    print(len(update))
    df.at[update, 'Gender'] = np.nan # changable
    t = df[['NameValue', 'Gender', 'AGE', 'Binary Age Code', 'Age Code', 'Department', 'Sample', 'GeoCode']]
    t.to_csv('0.Missing.csv', index=False)


    # In[50]:


    dfnm = df[['Gender', 'AGE', 'Department', 'Sample']]


    # ### Generate iMICE result

    # In[51]:


    i_mice_result = iMICE.iMICE(verbose=False, init_fill_method="median", impute_type="pmm", n_imputations=7).complete(np.matrix(dfnm))  # Here 7 is number of prediction for generating mean or median
    print(i_mice_result.shape)
    i_mice_result


    # ### Generate MICE result

    # In[52]:


    mice_result = MICE.MICE(verbose=False, init_fill_method="median", impute_type="pmm", n_imputations=7).complete(np.matrix(dfnm))  # Here 7 is number of prediction for generating mean or median
    print(mice_result.shape)
    mice_result


    # ### Generate KNN result

    # In[53]:


    knnImpute = KNN(k=7) # Here 7 is number of each cluster size
    knn_result = knnImpute.complete(dfnm.as_matrix())


    # ### Generate KMean result

    # In[54]:


    _,_,kmean_result = kmeans.kmeans_missing(dfnm.as_matrix(), n_clusters=7) # Here 7 is number of each cluster size


    # ### Create new DataFrame from iMICE result

    # In[55]:


    newdf = pd.DataFrame(i_mice_result, columns = ['Gender', 'AGE', 'Department', 'Sample'])
    predict = newdf['Gender'].map(lambda x: int(x)) # changable
    newdf['Name'] = df['NameValue']  
    newdf['Address'] = df['GeoCode']
    newdf = newdf[['Name', 'Gender', 'AGE', 'Department', 'Sample', 'Address']]

    newdf['Gender'] = newdf['Gender'].apply(lambda x: int(x))  # Normalize Gender
    newdf['AGE'] = newdf['AGE'].apply(lambda x: int(x))  # Normalize AGE
    newdf['Department'] = newdf['Department'].apply(lambda x: int(x))  # Normalize Department
    newdf['Sample'] = newdf['Sample'].apply(lambda x: int(x))  # Normalize Department

    newdf.to_csv('1.AfteriMICE2.csv', index=False)
    newdf.head()


    # ### Evaluation for iMICE

    # In[56]:


    print('Accuracy for iMICE: ')
    print(accuracy_score(actual, predict))
    print('Precision for iMICE: ')
    print(precision_score(actual, predict, average='weighted'))
    print('Recall for iMICE: ')
    print(recall_score(actual, predict, average='weighted'))
    print('f1 score for iMICE: ')
    print(f1_score(actual, predict, average='weighted'))


    # ### Create new DataFrame from MICE result

    # In[57]:


    ndf = pd.DataFrame(mice_result, columns = ['Gender', 'AGE', 'Department', 'Sample'])
    predict = ndf['Gender'].map(lambda x: int(x)) # changable
    ndf['Name'] = df['NameValue']
    ndf['Address'] = df['GeoCode']
    ndf = ndf[['Name', 'Gender', 'AGE', 'Department', 'Sample', 'Address']]

    ndf['Gender'] = ndf['Gender'].apply(lambda x: int(x))
    ndf['AGE'] = ndf['AGE'].apply(lambda x: int(x))
    ndf['Department'] = ndf['Department'].apply(lambda x: int(x))
    ndf['Sample'] = ndf['Sample'].apply(lambda x: int(x))

    ndf.to_csv('2.AfterMICE2.csv', index=False)
    ndf.head()


    # ### Evaluation for MICE

    # In[58]:


    print('Accuracy for MICE: ')
    print(accuracy_score(actual, predict))
    print('Precision for MICE: ')
    print(precision_score(actual, predict, average='weighted'))
    print('Recall for MICE: ')
    print(recall_score(actual, predict, average='weighted'))
    print('f1 score for MICE: ')
    print(f1_score(actual, predict, average='weighted'))


    # ### Create new DataFrame from KNN result

    # In[59]:


    newdf = pd.DataFrame(knn_result, columns = ['Gender', 'AGE', 'Department', 'Sample'])
    predict = newdf['Gender'].map(lambda x: int(x)) # changable
    newdf['Name'] = df['NameValue']
    newdf['Address'] = df['GeoCode']
    newdf = newdf[['Name', 'Gender', 'AGE', 'Department', 'Sample', 'Address']]

    newdf['Gender'] = newdf['Gender'].apply(lambda x: int(x))
    newdf['AGE'] = newdf['AGE'].apply(lambda x: int(x))
    newdf['Department'] = newdf['Department'].apply(lambda x: int(x))
    newdf['Sample'] = newdf['Sample'].apply(lambda x: int(x))

    newdf.to_csv('3.AfterKNN1.csv', index=False)
    newdf.head()


    # ### Evaluation for KNN

    # In[60]:


    print('Accuracy for KNN: ')
    print(accuracy_score(actual, predict))
    print('Precision for KNN: ')
    print(precision_score(actual, predict, average='weighted'))
    print('Recall for KNN: ')
    print(recall_score(actual, predict, average='weighted'))
    print('f1 score for KNN: ')
    print(f1_score(actual, predict, average='weighted'))


    # ### Create new DataFrame from KMeans result

    # In[61]:


    newdf = pd.DataFrame(kmean_result, columns = ['Gender', 'AGE', 'Department', 'Sample'])
    predict = newdf['Gender'].map(lambda x: int(x)) # changable
    newdf['Name'] = df['NameValue']
    newdf['Address'] = df['GeoCode']
    newdf = newdf[['Name', 'Gender', 'AGE', 'Department', 'Sample', 'Address']]

    newdf['Gender'] = newdf['Gender'].apply(lambda x: int(x))
    newdf['AGE'] = newdf['AGE'].apply(lambda x: int(x))
    newdf['Department'] = newdf['Department'].apply(lambda x: int(x))
    newdf['Sample'] = newdf['Sample'].apply(lambda x: int(x))

    newdf.to_csv('4.AfterKMean1.csv', index=False)
    newdf.head()


    # ### Evaluation for KMean

    # In[62]:


    print('Accuracy for KMean: ')
    print(accuracy_score(actual, predict))
    print('Precision for KMean: ')
    print(precision_score(actual, predict, average='weighted'))
    print('Recall for KMean: ')
    print(recall_score(actual, predict, average='weighted'))
    print('f1 score for KMean: ')
    print(f1_score(actual, predict, average='weighted'))
