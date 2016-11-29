#%%

import os
import pandas as pd
import datetime as dt



importDataPath = u'C:\\NHLData'
importFileNames = [x for x in os.listdir(importDataPath) if x.endswith(u'.xls')]



# create empty data frame for export data
allExportData = pd.DataFrame()



for i, fileName in enumerate(importFileNames):
    
    season = fileName[-11:-4]
    seasonStartYear = int(season[:4])
    
    
    # read data from file, remove spaces from headers
    fileNameWithPath = u'%s\\%s' % (importDataPath, fileName)
    fileData = pd.read_excel(fileNameWithPath)
    fileData.columns = [x.replace(u' ', u'') for x in fileData.columns.values]
    

    # create data frame with parameters
    exportData = fileData.loc[:, [u'LastName', u'FirstName']].copy()
    
    
    
    # different date format in different excel files
    if season == u'2015-16':
        exportData.loc[:, u'DateOfBirth'] = [dt.datetime.strptime(x, u"%Y-%m-%d") for x in fileData.DOB]
    else:
        exportData.loc[:, u'DateOfBirth'] = [dt.datetime.strptime(x, u"%b %d '%y") for x in fileData.DOB]
    

    # construct id string from namn and date of birth to compare between seasons
    exportData.loc[:, u'IdStr'] = [(u'%s_%s_%s' % (ln, fn, dt.datetime.strftime(dob, u"%Y-%m-%d")))
        for ln, fn, dob in zip(exportData.LastName, exportData.FirstName, exportData.DateOfBirth)]
    
    
    
    print i, fileName, season,
    
    if len(set(exportData.IdStr)) != len(exportData):
        print u': Warning: ID string is not unique!'
    else:
        print u': ID string is unique!'
    


    # add columns to data frame
    exportData.loc[:, u'Season'] = season
    exportData.loc[:, u'Age'] = [seasonStartYear - x.year for x in exportData.DateOfBirth]
    
    
    
    # different team column header in different excel files
    if season == u'2011-12':
        exportData.loc[:, u'EndTeam'] = fileData.Team
    else:
        exportData.loc[:, u'EndTeam'] = fileData.EndTeam
    


    # positions
    exportData.loc[:, u'MainPos'] = [x.split(u'/')[0] for x in fileData.Pos]
    
    exportData.loc[:, u'IsCenter']    = [int(x) for x in (exportData.MainPos == u'C')]
    exportData.loc[:, u'IsLeftWing']  = [int(x) for x in (exportData.MainPos == u'LW')]
    exportData.loc[:, u'IsRightWing'] = [int(x) for x in (exportData.MainPos == u'RW')]
    
    
    
    # countries
    exportData.loc[:, u'Country'] = fileData.Ctry
    
    exportData.loc[:, u'IsCanadian'] = [int(x) for x in (fileData.Ctry == u'CAN')]
    exportData.loc[:, u'IsAmerican'] = [int(x) for x in (fileData.Ctry == u'USA')]
    exportData.loc[:, u'IsSwedish']  = [int(x) for x in (fileData.Ctry == u'SWE')]
    exportData.loc[:, u'IsRussian']  = [int(x) for x in (fileData.Ctry == u'RUS')]
    exportData.loc[:, u'IsCzech']    = [int(x) for x in (fileData.Ctry == u'CZE')]
    exportData.loc[:, u'IsFinnish']  = [int(x) for x in (fileData.Ctry == u'FIN')]
    

    
    # other parameters
    exportData.loc[:, u'GamesPlayed']    = fileData.GP
    exportData.loc[:, u'Goals']          = fileData.G
    exportData.loc[:, u'GoalsPerGame']   = [g / float(gp) if gp > 0 else 0.0 for gp, g in zip(fileData.GP, fileData.G)]
    exportData.loc[:, u'Assists']        = fileData.A
    exportData.loc[:, u'AssistsPerGame'] = [a / float(gp) if gp > 0 else 0.0 for gp, a in zip(fileData.GP, fileData.A)]
    
    exportData.loc[:, u'ShirtNumber'] = fileData.loc[:, u'#']
    exportData.loc[:, u'ShirtNumberLargerThanForty'] = [1 if x > 40 else 0 for x in exportData.ShirtNumber]
    
    
    # append
    allExportData = allExportData.append(exportData, ignore_index=True)
    



#%%

# export to file

exportDataPath = '%s\\StandardizedData' % importDataPath

allExportData.to_csv((u'%s\\nhl_data.csv' % exportDataPath), index=False, encoding=u'utf-8')


#%%

# console output

print u'\nEXPORT DATA:\n\n', allExportData.head()

#print '\nROWS PER SEASON:\n\n', allExportData.Season.value_counts()




    


