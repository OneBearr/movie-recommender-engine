import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})     

# read the .txt in
rawTrainData = np.loadtxt('data-files/train.txt', dtype = int)
rawTest5Data = np.loadtxt('data-files/test5.txt', dtype = int)
rawTest10Data = np.loadtxt('data-files/test10.txt', dtype = int)
rawTest20Data = np.loadtxt('data-files/test20.txt', dtype = int)

# convert raw data to a 501*1001 array (users vs movies), 
# the first column is the average rating for each user, the first row is the average rating for each movie
# 1-200 for training users, 201-300 for test5 users, 301-400 for test10 users, 401-500 for test20 users
dataArr = np.zeros((501, 1001))         # dataArr[u][m] means user u's rating for movie m

def form_table():
    for i in rawTrainData:              # user 1-200
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r
        
    for i in rawTest5Data:              # user 201-300
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # need to be predicted

    for i in rawTest10Data:             # user 301-400
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # need to be predicted

    for i in rawTest20Data:             # user 401-500
        u, m, r = i[0], i[1], i[2]
        dataArr[u][m] = r if r > 0 else -1     # need to be predicted
    
    for u in range(1, 501):             # store the ave rating for each user             
        ratings = [dataArr[u][m] for m in range(1, 1001) if dataArr[u][m] > 0]
        dataArr[u][0] = sum(ratings) / len(ratings)
            
    for m in range(1, 1001):            # store the ave rating for each movie
        ratings = [dataArr[u][m] for u in range(1, 501) if dataArr[u][m] > 0]
        dataArr[0][m] = sum(ratings) / len(ratings) if ratings else -1  # -1 if nobody rated this movie         

def predict(movie_id, new_user_id):    # calculate the predicted rating by similarity weights and neighbors rating
    res, p = 0, 0.635
    if dataArr[0][movie_id] > 0:        # take portion of movie ave rating
        res = dataArr[0][movie_id] * p +  dataArr[new_user_id][0] * (1 - p)
    else:                                 # if the movie ave rating is 0, take the user ave rating
        res = dataArr[new_user_id][0]
    return round(res)       # res is always btw 1-5, since user ave and movie ave are always btw 1-5

form_table()

f1 = open("own-algo/own_algo_result5.txt", "w")       # predictions for result5.txt
for new_user in range(201,301):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, new_user)
            f1.write(str(new_user)+" "+ str(movie_id)+" "+ str(pred)+"\n")
f1.close()
print('Result5 Prediction Done')

f2 = open("own-algo/own_algo_result10.txt", "w")      # predictions for result10.txt
for new_user in range(301,401):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, new_user)
            f2.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f2.close()
print('Result10 Prediction Done')

f3 = open("own-algo/own_algo_result20.txt", "w")      # predictions for result20.txt
for new_user in range(401,501):
    for movie_id in range(1, 1001):
        if dataArr[new_user][movie_id] == -1:
            pred = predict(movie_id, new_user)
            f3.write(str(new_user)+" "+str(movie_id)+" "+str(pred)+"\n")
f3.close()
print('Result20 Prediction Done')
