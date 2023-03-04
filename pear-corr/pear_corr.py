import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# read the .txt in
rawTrainData = np.loadtxt('DataFiles/train.txt', dtype = int)
rawTest5Data = np.loadtxt('DataFiles/test5.txt', dtype = int)
rawTest10Data = np.loadtxt('DataFiles/test10.txt', dtype = int)
rawTest20Data = np.loadtxt('DataFiles/test20.txt', dtype = int)

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
        if (r == 0):
            dataArr[u][m] = -1      # need to be predicted
        else:
            dataArr[u][m] = r

    for i in rawTest10Data:         # user 301-400
        u, m, r = i[0], i[1], i[2]
        if (r == 0):
            dataArr[u][m] = -1      # need to be predicted
        else:
            dataArr[u][m] = r

    for i in rawTest20Data:         # user 401-500
        u, m, r = i[0], i[1], i[2]
        if (r == 0):
            dataArr[u][m] = -1      # need to be predicted
        else:
            dataArr[u][m] = r
   
    for m in range(1, 1001):        # store the ave rating for each movie
        rate_num = 0
        rate_sum = 0
        for u in range(1, 501):
            if dataArr[u][m] > 0:
                rate_num += 1
                rate_sum += dataArr[u][m]
        if rate_num == 0:
            dataArr[0][m] = -1      # nobody rated this movie
        else:
            dataArr[0][m] = rate_sum / rate_num
    
    for u in range(1, 501):         # store the ave rating for each user
        rate_num = 0
        rate_sum = 0
        for m in range(1, 1001):
            if dataArr[u][m] > 0:
                rate_num += 1
                rate_sum += dataArr[u][m]
        dataArr[u][0] = rate_sum / rate_num  

def calculate_PearCor_similarities(new_user_id):     # calculate the sims btw the new user and 1-200 old users
    new_user = dataArr[new_user_id]
    sims = []
    sims.append(0.0)                # for dummy user0 who does not exist
    for i in range(1, 201):         # calculate the similarities for each old user 1-200
        old_user = dataArr[i]
        nume, sqrN, sqrO, cnt = 0.00, 0.00, 0.00, 0
        for j in range(1, 1001):    # from all the common rated movies
            if new_user[j] > 0 and old_user[j] > 0:
                nume += (new_user[j] - new_user[0]) * (old_user[j] - old_user[0])
                sqrN += (new_user[j] - new_user[0]) ** 2
                sqrO += (old_user[j] - old_user[0]) ** 2
                cnt += 1
        
        if cnt <= 1 or sqrN == 0 or sqrO == 0:        # no common rated movie or only one common rated movie
            sims.append(0.00)
        else:
            sim = nume / (np.sqrt(sqrN) * np.sqrt(sqrO))
            sims.append(sim)        # the similarity of the old_user i and the new_user
                
    return sims

def find_decreasing_sims_indexes(sims, len):    # find k biggest absolute values' indexes in sims
    return np.argsort(np.abs(sims))[-len:][::-1]            

def predict(movie_id, sims, idxs, new_user_id):    # calculate the predicted rating by similarity weights and neighbors rating
    nume, deno, res = 0.00, 0.00, 0.00
    k = 0
    res = dataArr[new_user_id][0]
    for old_user_id in idxs:
        if dataArr[old_user_id][movie_id] > 0 and np.abs(sims[old_user_id]) > 0.5 and k < 10:
            nume += sims[old_user_id] * (dataArr[old_user_id][movie_id] - dataArr[old_user_id][0])
            deno += np.abs(sims[old_user_id])
            k += 1
    
    if deno != 0 and k > 0:
        res += nume / deno
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

f1 = open("PearCor/PearCor-result5.txt", "w")       # predictions for result5.txt
for new_user in range(201,301):
    sims = calculate_PearCor_similarities(new_user)         # 1x201 for sims
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f1.write(str(new_user)+" "+ str(movie_id)+" "+ str(pred)+"\n")
f1.close()

f2 = open("PearCor/PearCor-result10.txt", "w")      # predictions for result10.txt
for new_user in range(301,401):
    sims = calculate_PearCor_similarities(new_user)         # 1x201 for sims
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f2.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f2.close()

f3 = open("PearCor/PearCor-result20.txt", "w")      # predictions for result20.txt
for new_user in range(401,501):
    sims = calculate_PearCor_similarities(new_user)         # 1x201 for sims
    idxs = find_decreasing_sims_indexes(sims, len(sims))    # 1x201 for idxs
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, sims, idxs, new_user)
            f3.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f3.close()
