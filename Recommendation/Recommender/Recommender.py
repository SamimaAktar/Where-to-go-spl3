import numpy as np
import math
class Recommender:
    def __init__(self):
        self.user_based=user_based
        self.friend_based=friend_based
        self.location_based=location_based
        self.all_uids=all_uids
        self.all_lids=all_lids
        self.training_matrix=training_matrix
        self. poi_coos=poi_coos
    def predict(self,uid):
        alpha = 0.1
        beta = 0.1
        all_uids=self.all_uids
        all_lids=self.all_lids
        training_matrix= self.training_matrix
        if uid in all_uids: 
            U_scores = self.normalize([self.user_based.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            S_scores = self.normalize([self.friend_based.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])    
            G_scores = self.normalize([self.location_based.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[:5]
            
            return predicted
    def normalize(self,scores):
        max_score = max(scores)
        if not max_score == 0:
            scores = [s / max_score for s in scores]
        return scores
    def getLocationInfo(self):
        return self.poi_coos


class FriendBasedCF(object):
    def predict(self, i, j):
        if i in self.social_proximity:
            numerator = np.sum([(self.eta * jf + (1 - self.eta) * jc) * self.check_in_matrix[k, j]
                                for k, jf, jc in self.social_proximity[i]])
            return numerator
        return 0.0


class UserBasedCF(object):
    def predict(self, i, j):
        return self.rec_score[i][j]

class PowerLaw(object):
    def dist(self,loc1, loc2):
        lat1, long1 = loc1[0], loc1[1]
        lat2, long2 = loc2[0], loc2[1]
        if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
            return 0.0
        degrees_to_radians = math.pi/180.0
        phi1 = (90.0 - lat1)*degrees_to_radians
        phi2 = (90.0 - lat2)*degrees_to_radians
        theta1 = long1*degrees_to_radians
        theta2 = long2*degrees_to_radians
        cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
               math.cos(phi1)*math.cos(phi2))
        arc = math.acos( cos )
        earth_radius = 6371
        return arc * earth_radius
    def pr_d(self, d):
        d = max(0.01, d)
        return self.a * (d ** self.b)
    def predict(self, uid, lj):
        lj = self.poi_coos[lj]
        return np.prod([self.pr_d(self.dist(self.poi_coos[li], lj)) for li in self.visited_lids[uid]])
   


