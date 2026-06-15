import numpy as np
import os
import matplotlib.pyplot as plt
class BB:
    def __init__(self,probs,seed=10):
        self.probs=np.array(probs)
        self.rng=np.random.default_rng(seed)
    def pull(self,arm):
        rdn=self.rng.random()
        if(rdn<=self.probs[arm]):
            return 1
        else:
            return 0
class RA:
    def __init__(self,n_arms,seed=2):
        self.n_arms=n_arms
        self.rng=np.random.default_rng(seed)
        self.count=np.zeros(n_arms)
        self.q_values=np.zeros(n_arms)
    def select_arm(self):
        random_number=self.rng.integers(self.n_arms)
        return random_number
    def update(self,arms,reward):
        self.count[arms]+=1
        n=self.count[arms]
        self.q_values[arms]+=(reward-self.q_values[arms])/n
RAA=RA(3)
bandit=BB([0.5,0.6,0.8])
regret=[]
best_mean=np.max(bandit.probs)
cumulative_regret=0
for i in range(1000):
    sss=RAA.select_arm()
    chooes_mean=bandit.probs[sss]
    lose=best_mean-chooes_mean
    cumulative_regret+=lose
    regret.append(cumulative_regret)
    ss=bandit.pull(sss)
    RAA.update(sss,ss)
os.makedirs("result_version1", exist_ok=True)
plt.plot(regret)
plt.xlabel("Round")
plt.ylabel("cumulative regret")
plt.title("Random Agent Regret Curve--random")
plt.savefig("result_version1/random__.png", dpi=200, bbox_inches="tight")  
