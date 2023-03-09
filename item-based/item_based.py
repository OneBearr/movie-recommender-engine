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
dataArr = np.zeros((501, 1001))

def form_table():
    for i in rawTrainData:          # user 1-200
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r
        
    for i in rawTest5Data:          # user 201-300
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted

    for i in rawTest10Data:         # user 301-400
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted

    for i in rawTest20Data:         # user 401-500
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # -1 needs to be predicted
   
    for m in range(1, 1001):        # store the ave rating for each movie
        ratings = [dataArr[u][m] for u in range(1, 501) if dataArr[u][m] > 0]
        sum, cnt = np.sum(ratings), len(ratings)
        dataArr[0][m] = sum / cnt if ratings else -1  # -1 if nobody rated this movie
    
    for u in range(1, 501):         # store the ave rating for each user
        ratings = [dataArr[u][m] for m in range(1, 1001) if dataArr[u][m] > 0]
        sum, cnt = np.sum(ratings), len(ratings)
        dataArr[u][0] = sum / cnt

def calculate_movie_similarities(movie_id):
    sims = [0.00]                    # for dummy movie0 who does not exist
    for m in range(1, 1001):
        nume, sqrN, sqrO, cnt = 0.00, 0.00, 0.00, 0
        for u in range(1, 501):
            if dataArr[u][movie_id] > 0 and dataArr[u][m] > 0:
                nume = (dataArr[u][movie_id] - dataArr[u][0]) * (dataArr[u][m] - dataArr[u][0])
                sqrN = (dataArr[u][movie_id] - dataArr[u][0]) ** 2
                sqrO = (dataArr[u][m] - dataArr[u][0]) ** 2
                cnt += 1
        if cnt <= 1 or sqrN == 0 or sqrO == 0:
            sims.append(0.00)
        else:
            sim = nume / (np.sqrt(sqrN) * np.sqrt(sqrO))
            sims.append(sim)
    return sims

def find_decreasing_sims_indexes(sims, len):    # find k biggest absolute values' indexes in sims
    return np.argsort(np.abs(sims))[-len:][::-1]            

def item_based_predict(sims, idxs, new_user_id, movie_id, k):
    nume, deno, res = 0.00, 0.00, 0.00
    n = 0
    res = dataArr[new_user_id][0]
    for m in idxs:
        if dataArr[new_user_id][m] > 0 and np.abs(sims[m]) > 0.5 and n < k:
            nume += sims[m] * (dataArr[new_user_id][m] - dataArr[new_user_id][0])
            deno += np.abs(sims[m])
            n += 1    
    if deno != 0 and n > 0:
        res += nume / deno
    elif dataArr[0][movie_id] > 0:
        res = dataArr[0][movie_id]
    else:
        res = dataArr[new_user_id][0]    
    if res < 1:
        res = 1.0
    elif res > 5:
        res = 5.0
    return round(res)

form_table()

predArr = np.zeros((501, 1001))         # array for storing prediction results
k = 0
for movie_id in range(1, 1001):
    sims = calculate_movie_similarities(movie_id)             # 1x1001 for sims
    idxs = find_decreasing_sims_indexes(sims, len(sims))      # 1x1001 for idxs
    for new_user in range (201, 501):
        if dataArr[new_user][movie_id] == -1:
            if 200 <= new_user < 301:
                k = 5
            elif 300 <= new_user < 401:
                k = 10
            else:
                k = 20
            pred = item_based_predict(sims, idxs, new_user, movie_id, k)
            predArr[new_user][movie_id] = pred
    
    
f1 = open("item-based/item_based_result5.txt", "w")       # predictions for result5.txt
for new_user in range(201,301):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            f1.write(str(new_user)+" "+ str(movie_id)+" "+ str(predArr[new_user][movie_id])+"\n")
f1.close()
print('Result5 Prediction Done')

f2 = open("item-based/item_based_result10.txt", "w")       # predictions for result10.txt
for new_user in range(301,401):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            f2.write(str(new_user)+" "+ str(movie_id)+" "+ str(predArr[new_user][movie_id])+"\n")
f2.close()
print('Result10 Prediction Done')

f3 = open("item-based/item_based_result20.txt", "w")       # predictions for result20.txt
for new_user in range(401,501):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            f3.write(str(new_user)+" "+ str(movie_id)+" "+ str(predArr[new_user][movie_id])+"\n")
f3.close()
print('Result20 Prediction Done')
print('All Prediction Done')