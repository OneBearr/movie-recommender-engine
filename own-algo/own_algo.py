import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# read the .txt in
rawTrainData = np.loadtxt('data-files/train.txt', dtype = int)
rawTest5Data = np.loadtxt('data-files/test5.txt', dtype = int)
rawTest10Data = np.loadtxt('data-files/test10.txt', dtype = int)
rawTest20Data = np.loadtxt('data-files/test20.txt', dtype = int)

# convert raw data to a 501*1001 array
# the first column is the average rating for each user, the first row is the average rating for each movie
# 1-200 for training users, 201-300 for test5 users, 301-400 for test10 users, 401-500 for test20 users
dataArr = np.zeros((503, 1003))

iufArr = np.zeros(1001)
iufRateArr = np.zeros((501, 1001))

def form_table():
    for i in rawTrainData:              # user 1-200
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r
        
    for i in rawTest5Data:              # user 201-300
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted

    for i in rawTest10Data:             # user 301-400
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted

    for i in rawTest20Data:             # user 401-500
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted
    
    for u in range(1, 501):             # store the ave rating for each user             
        ratings = [dataArr[u][m] for m in range(1, 1001) if dataArr[u][m] > 0]
        sum, cnt = np.sum(ratings), len(ratings)
        dataArr[u][0] = sum / cnt
        
        dataArr[u][1001], dataArr[u][1002] = sum, cnt
    
    for m in range(1, 1001):            # store the ave rating for each movie
        ratings = [dataArr[u][m] for u in range(1, 501) if dataArr[u][m] > 0]
        sum, cnt = np.sum(ratings), len(ratings)
        dataArr[0][m] = sum / cnt if ratings else -1  # -1 if nobody rated this movie
        dataArr[501][m], dataArr[502][m] = sum, cnt
        iufArr[m] = np.log(500 / len(ratings)) if ratings else 0    # store the iuf for each movie
    
    for u in range(1, 501):             # store IUFed movie ratings for each user
        for m in range(1, 1001):
            if dataArr[u][m] > 0:  
                iufRateArr[u][m] = dataArr[u][m] * iufArr[m]
    
    for u in range(1, 501):             # store the IUFed ave rating for each user
        ratings = [iufRateArr[u][m] for m in range(1, 1001) if iufRateArr[u][m] > 0]
        iufRateArr[u][0] = np.sum(ratings) / len(ratings)
    
    for m in range(1, 1001):            # store the IUFed ave rating for each movie
        ratings = [iufRateArr[u][m] for u in range(1, 501) if iufRateArr[u][m] > 0]
        iufRateArr[0][m] = np.sum(ratings) / len(ratings) if ratings else -1   # -1 if nobody rated this movie
      
def calculate_PearCorIUF_similarities(new_user_id):     # calculate the sims btw the new user and 1-200 old users
    new_user = dataArr[new_user_id]
    new_user_iuf = iufRateArr[new_user_id]
    sims = [0.00]                    # for dummy user0 who does not exist
    for i in range(1, 201):         # calculate the similarities for each old user 1-200
        old_user = dataArr[i]       # old_user_iuf = iufRateArr[i]
        nume, sqrN, sqrO, cnt = 0.00, 0.00, 0.00, 0
        for j in range(1, 1001):    # from all the common rated movies
            if new_user[j] > 0 and old_user[j] > 0:
                nume += (new_user_iuf[j] - new_user[0]) * (new_user_iuf[j] - old_user[0])
                sqrN += (new_user_iuf[j] - new_user[0]) ** 2
                sqrO += (new_user_iuf[j] - old_user[0]) ** 2
                cnt += 1
        if cnt <= 1 or sqrN == 0 or sqrO == 0:        # no common rated movie or only one common rated movie
            sims.append(0.00)
        else:
            sims.append(nume / (np.sqrt(sqrN) * np.sqrt(sqrO)))        # the similarity of the old_user i and the new_user 
    # apply Case Amplification on sims(weights)       
    return np.power(np.abs(sims), 1.5) * sims

def find_decreasing_sims_indexes(sims, len):    # find k biggest absolute values' indexes in sims
    return np.argsort(np.abs(sims))[-len:][::-1]            

def predict(movie_id, sims, idxs, new_user_id):    # calculate the predicted rating by similarity weights and neighbors rating
    nume, deno, k= 0.00, 0.00, 0 
    res = dataArr[new_user_id][0]
    for old_user_id in idxs:
        if dataArr[old_user_id][movie_id] > 0 and np.abs(sims[old_user_id]) > 0.8 and k < 17:  
            nume += sims[old_user_id] * (dataArr[old_user_id][movie_id] - dataArr[old_user_id][0])
            deno += np.abs(sims[old_user_id])
            k += 1
    nume += dataArr[0][movie_id] - dataArr[new_user_id][0]  # add the movie ave rating as well 
    if deno != 0 and k > 0:
        res += nume / (deno+1)          # since we added the movie ave rating
    elif dataArr[0][movie_id] > 0:        # deno is 0(k is 0), there is no similar user, take the movie ave rating
        res = dataArr[0][movie_id]
    else:                                 # if the movie ave rating is 0, take the user ave rating
        res = dataArr[new_user_id][0]
    if res < 1:
        res = 1.0
    elif res > 5:
        res = 5.0
    return round(res)

form_table()

f1 = open("own-algo/own_algo_result5.txt", "w")       # predictions for result5.txt
for new_user in range(201,301):
    sims = calculate_PearCorIUF_similarities(new_user)         # 1x201 for sim
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f1.write(str(new_user)+" "+ str(movie_id)+" "+ str(pred)+"\n")
f1.close()
print('Result5 Prediction Done')

f2 = open("own-algo/own_algo_result10.txt", "w")      # predictions for result10.txt
for new_user in range(301,401):
    sims = calculate_PearCorIUF_similarities(new_user)         # 1x201 for sim
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f2.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f2.close()
print('Result10 Prediction Done')

f3 = open("own-algo/own_algo_result20.txt", "w")      # predictions for result20.txt
for new_user in range(401,501):
    sims = calculate_PearCorIUF_similarities(new_user)         # 1x201 for sim
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f3.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f3.close()
print('Result20 Prediction Done')
print('All Prediction Done')