
#%%

import pandas as pd
import numpy as np
from sklearn import linear_model, cross_validation

import itertools
import matplotlib.pyplot as plt


## format output
import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 7.0)
matplotlib.rcParams['font.size'] = 18.0
matplotlib.rcParams['lines.linewidth'] = 3.0
matplotlib.rcParams['lines.markersize'] = 10.0




importDataPath = u'C:\\NHLData\\StandardizedData'

nhlData = pd.read_csv(u'%s\\%s' % (importDataPath, u'nhl_data.csv'))



modelSeason = u'2013-14'
responseSeason = u'2014-15'


minimumNumberOfGames = 10

df1 = nhlData[(nhlData.Season == modelSeason) & (nhlData.GamesPlayed >= minimumNumberOfGames)]
df2 = nhlData[nhlData.Season == responseSeason].loc[:, [u'IdStr', u'Goals']]

modelData = df1.merge(df2, on=u'IdStr', how=u'inner', suffixes=(u'', u'NextSeason'))


print '\nMODEL DATA:\n\n', modelData.head()


#%%


## response variable

y = np.array(modelData.GoalsNextSeason, dtype=np.float64)

## simple model, linear regression with one regressor


x1 = np.array(modelData.Goals, dtype=np.float64)
X = np.vstack([x1]).T

simpleModel = linear_model.LinearRegression()
simpleModel.fit(X, y)

yPredSimpleModel = simpleModel.predict(X)
epsSimpleModel = y - yPredSimpleModel

aHat = simpleModel.intercept_
bHat = simpleModel.coef_[0]

def predictNumberOfGoalsSimpleModel(x):
    return aHat + bHat*x


yMax = np.max([np.max(yPredSimpleModel), np.max(y)])

print u'\n*** SIMPLE MODEL ***'

plt.plot(yPredSimpleModel, y, u'bo')
plt.plot([0.0, yMax+1], [0.0, yMax+1], u'r--')

plt.xlabel(u'Predicted, simple model')
plt.ylabel(u'Actual')
plt.title(u'Number of goals next season')
plt.show()

print u'\nGoalsNextSeason = %.2f + %.2f*Goals + error term' % (simpleModel.intercept_, simpleModel.coef_[0])


del x1, X, yMax

#%%


## better model (?), lasso to choose regressors

regressors = [
    u'GamesPlayed'
    , u'GoalsPerGame'
    , u'AssistsPerGame'
    , u'Age'
    , u'IsCenter'
    , u'IsRightWing'
    , u'IsLeftWing'
    , u'IsCanadian'
    , u'IsAmerican'
    , u'IsSwedish'
    , u'IsRussian'
    , u'IsCzech'
    , u'IsFinnish'
    , u'ShirtNumberLargerThanForty'
    ]


# convert to numpy array (not matrix)
Z = modelData.loc[:, regressors].as_matrix().astype(np.float64)


X = (Z - Z.mean(axis=0)) / Z.std(axis=0)


# cross validation to choose regularization parameter (alpha) in lasso

alphas, _, _ = linear_model.lasso_path(X, y)
negLogAlphas = -np.log(alphas)


randomState = 235982     # to get the same result each run
numberOfObservations = len(y)
numberOfAlphas = len(alphas)
numberOfFolds = 5


kFold = cross_validation.KFold(
    numberOfObservations
    , n_folds=numberOfFolds
    , shuffle=True
    , random_state=randomState)


squaredErrors = np.empty([numberOfAlphas, numberOfObservations])

print u'\n*** LASSO MODEL: find regularization parameter ***\n'

lasso = linear_model.Lasso(alpha=1.0)

for i, idx in enumerate(kFold):
    
    trainIdx, testIdx = idx
    
    XTrain = X[trainIdx]
    yTrain = y[trainIdx]
        
    XTest = X[testIdx]
    yTest = y[testIdx]
    
    for j, alpha in enumerate(alphas):
        lasso.set_params(alpha=alpha)
        lasso.fit(XTrain, yTrain)
        
        yPred = lasso.predict(XTest)
        
        squaredErrors[j, testIdx] = (yTest - yPred)**2
        

## choose alpha with minimal mean squared error
    
meanSquaredErrors = squaredErrors.mean(axis=1)
mseMin = np.min(meanSquaredErrors)
mseIdx = np.argmin(meanSquaredErrors)


chosenAlpha = alphas[mseIdx]
negLogChosenAlpha = negLogAlphas[mseIdx]


# get lasso path coefficients

_, allCoefs, _ = linear_model.lasso_path(X, y, alphas=alphas)

# plot mean squared errors

plt.plot(negLogAlphas, meanSquaredErrors, u'b-')
plt.plot([negLogChosenAlpha], [mseMin], u'ro')
plt.ylim([mseMin*0.99, mseMin*1.10])
plt.xlabel(u'-log(regularization parameter)')
plt.ylabel(u'test mean squared error')
plt.show()

print u'\n-log(regularization parameter) = %.2f' % negLogChosenAlpha


del Z, X, randomState, numberOfObservations
del numberOfAlphas, numberOfFolds, kFold, squaredErrors, lasso
del meanSquaredErrors, mseMin





print u'\n*** LASSO MODEL: plot lasso paths ***\n'

# plot lasso paths

colorList = [u'blue', u'red', u'green', u'cyan', u'magenta', u'black', u'orange']
nColors = len(colorList)
colors = itertools.cycle(colorList)
lineStyles = itertools.cycle([u'solid']*nColors + [u'dashed']*nColors + [u'dashdot']*nColors)


regressorsAndLassoCoefs = [(r, coefs[mseIdx], coefs) for r, coefs in zip(regressors, allCoefs)]
regressorsAndLassoCoefs.sort(key=lambda x : -np.abs(x[1]))



for rlc, cl, ls in zip(regressorsAndLassoCoefs, colors, lineStyles):
    regressor = rlc[0]
    chosenCoef = rlc[1]
    coefs = rlc[2]
    plt.plot(negLogAlphas, coefs, c=cl, linestyle=ls)
    print u'%26s  %7s  %7s  %6.2f' % (regressor, cl, ls, chosenCoef)



plt.plot([negLogChosenAlpha, negLogChosenAlpha]
            , [np.floor(np.min(allCoefs)), np.ceil(np.max(allCoefs))], u'k--')
plt.xlabel(u'-log(regularization parameter)')
plt.ylabel(u'coefficient')
plt.show()


chosenRegressors = [x[0] for x in regressorsAndLassoCoefs if x[1] != 0]

del colorList, nColors, colors, lineStyles, regressorsAndLassoCoefs



#%%

print u'\n*** LASSO MODEL: final model ***\n'


Z =  modelData.loc[:, chosenRegressors].as_matrix().astype(np.float64)

ZMean = Z.mean(axis=0)
ZStd = Z.std(axis=0)

X = (Z - ZMean) / ZStd


finalModel = linear_model.LinearRegression()
finalModel.fit(X, y)

yPredLassoModel = finalModel.predict(X)
epsLassoModel = y - yPredLassoModel

yMax = np.max([np.max(yPredLassoModel), np.max(y)])

plt.plot(yPredLassoModel, y, u'bo')
plt.plot([0.0, yMax+1], [0.0, yMax+1], u'r--')

plt.xlabel(u'Predicted, lasso model')
plt.ylabel(u'Actual')
plt.title(u'Number of goals next season')
plt.show()

chosenRegressorsAndCoefs = [(r,c) for r,c in zip(chosenRegressors, finalModel.coef_)]

print '\n                 REGRESSOR  COEFFICIENT'
for r,c in chosenRegressorsAndCoefs:
    print u'%26s  %11.2f' % (r, c)

    
alphaHat = finalModel.intercept_
betaHat = finalModel.coef_

theta0 = alphaHat - np.sum(betaHat * ZMean / ZStd)
theta1 = betaHat / ZStd


def predictNumberOfGoalsLassoModel(z):
    return theta0 + np.dot(theta1, z)


del Z, ZMean, ZStd, X, yMax, alphaHat, betaHat

#%%

print u'\n*** TEST : mean squared errors ***\n'

## test models on next season

testModelSeason = u'2014-15'
testResponseSeason = u'2015-16'


df1 = nhlData[(nhlData.Season == testModelSeason) & (nhlData.GamesPlayed >= minimumNumberOfGames)]
df2 = nhlData[nhlData.Season == testResponseSeason].loc[:, [u'IdStr', u'Goals']]

testData = df1.merge(df2, on=u'IdStr', how=u'inner', suffixes=(u'', u'NextSeason'))


gTest = np.array(testData.Goals)
ZTest = testData.loc[:, chosenRegressors].as_matrix().astype(np.float64)


testData.loc[:, u'GoalsPredictedSimpleModel'] = [predictNumberOfGoalsSimpleModel(x) for x in gTest]
testData.loc[:, u'GoalsPredictedLassoModel'] = np.apply_along_axis(predictNumberOfGoalsLassoModel, 1, ZTest)



goalsNextSeason = np.array(testData.GoalsNextSeason)
goalsPredictedSimpleModel = np.array(testData.GoalsPredictedSimpleModel)
goalsPredictedLassoModel = np.array(testData.GoalsPredictedLassoModel)

epsSimpleModel = goalsNextSeason - goalsPredictedSimpleModel
epsLassoModel = goalsNextSeason - goalsPredictedLassoModel

standardizedEpsSimpleModel = (epsSimpleModel - np.mean(epsSimpleModel)) / np.std(epsSimpleModel)
standardizedEpsLassoModel = (epsLassoModel - np.mean(epsLassoModel)) / np.std(epsLassoModel)

standardizedEpsMax = np.max([np.max(np.abs(standardizedEpsSimpleModel)), np.max(np.abs(standardizedEpsLassoModel)) ])



mseSimpleModel = np.mean(epsSimpleModel**2)
mseLassoModel = np.mean(epsLassoModel**2)


## scatter plots
pltLim = [-3, 53]
epsLim = [-np.ceil(standardizedEpsMax), np.ceil(standardizedEpsMax)]


plt.subplot(2,2,1)
plt.plot(goalsPredictedSimpleModel, goalsNextSeason, u'bo')
plt.plot(pltLim, pltLim, u'r--')
plt.xlim(pltLim)
plt.ylim(pltLim)
plt.ylabel(u'Goals ' + testResponseSeason)



plt.subplot(2,2,2)
plt.plot(goalsPredictedLassoModel, goalsNextSeason, u'bo')
plt.plot(pltLim, pltLim, u'r--')
plt.xlim(pltLim)
plt.ylim(pltLim)



plt.subplot(2,2,3)
plt.plot(goalsPredictedSimpleModel, standardizedEpsSimpleModel, u'bo')
plt.plot(pltLim, [0.0]*2,  u'r--')
plt.xlim(pltLim)
plt.ylim(epsLim)
plt.xlabel(u'Goals predicted, simple model')
plt.ylabel(u'Standardized errors')

plt.subplot(2,2,4)
plt.plot(goalsPredictedLassoModel, standardizedEpsLassoModel, u'bo')
plt.plot(pltLim, [0.0]*2,  u'r--')
plt.xlim(pltLim)
plt.ylim(epsLim)
plt.xlabel(u'Goals predicted, lasso model')


plt.show()



## display mean squared errors



print u'\nMEAN SQUARED ERRORS:\n'

print u'%12s: %.2f (%.2f)' % (u'simple model', mseSimpleModel, np.sqrt(mseSimpleModel))
print u'%12s: %.2f (%.2f)'  % (u'lasso model', mseLassoModel, np.sqrt(mseLassoModel))




del df1, df2, gTest, ZTest, goalsNextSeason, goalsPredictedSimpleModel
del goalsPredictedLassoModel, mseSimpleModel, mseLassoModel, pltLim



print u'\n*** TEST : top ten ***\n'


## display top ten

sortedTestData = testData.sort(u'GoalsNextSeason', ascending=False)
getRanking = {x:(i+1) for i, x in enumerate(sortedTestData.IdStr)}

testData.loc[:, u'Ranking'] = [getRanking[x] for x in testData.IdStr]
testData.loc[:, u'Name'] = [u'%s %s' % (fn, ln) for fn, ln in zip(testData.FirstName, testData.LastName)]


displayColumns = [u'Name']


n = 10

topTenSimpleModel = testData.loc[:, (displayColumns + [u'GoalsPredictedSimpleModel'
    , u'GoalsNextSeason', u'Ranking'])].sort(u'GoalsPredictedSimpleModel', ascending=False).head(n).copy()
topTenLassoModel = testData.loc[:, (displayColumns + [u'GoalsPredictedLassoModel'
    , u'GoalsNextSeason', u'Ranking'])].sort(u'GoalsPredictedLassoModel', ascending=False).head(n).copy()


topTenSimpleModel.index = np.arange(1, n + 1)
topTenLassoModel.index = np.arange(1, n + 1)




print u'\nTOP TEN SIMPLE MODEL:\n\n', topTenSimpleModel
print u'\nTOP TEN LASSO MODEL:\n\n', topTenLassoModel

del sortedTestData, getRanking, displayColumns, n


#%%

## data to Daniel

seasonDaniel = u'2015-16'
minimumNumberOfGamesDaniel = 10

danielData = nhlData[(nhlData.Season == seasonDaniel) & (nhlData.GamesPlayed >= minimumNumberOfGamesDaniel)].copy()


ZDaniel =  danielData.loc[:, chosenRegressors].as_matrix().astype(np.float64)

goalsPredictedDaniel = np.apply_along_axis(predictNumberOfGoalsLassoModel, 1, ZDaniel)

danielData.loc[:, u'GoalsPredictedLassoModel'] = [x if x > 0 else 0.0 for x in goalsPredictedDaniel]


exportDataPath = importDataPath

danielData.to_csv((u'%s\\daniel_data.csv' % exportDataPath), index=False, encoding=u'utf-8')



