from matplotlib import pyplot as plt
import numpy as np



x_values = np.linspace(-50, 100, 600)
mu_mix = [20,50,80]
sig_mix = [10,5,20]
mu_p = 50
sig_p = 30
total_samples = 10000

def gaussian(x, mu1, sig1):
    return np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.)))/(np.sqrt(2*np.pi)*sig1)


def get_scale_factor():
    sum = np.zeros(x_values.shape[0],)
    for i in range(len(mu_mix)):
        r =  (gaussian(x_values,mu_mix[i],sig_mix[i])/gaussian(x_values,mu_p,sig_p))
        sum+=r
    print("------value of M is "+str(np.max(sum))+" -------")
    # return(np.ceil(np.max(sum)))
    return((np.max(sum)))

def rejection_sampler(M,total_samples):
    samples = []
    cnt = 0
    while(len(samples)<total_samples):
        cnt+=1
        zCand = np.random.normal(50,30,1)
        q_zCand = gaussian(zCand,mu_p,sig_p)
        u = np.random.uniform(0,M*q_zCand,1)
        p_hat_zCand = 0
        for i in range(len(mu_mix)):
            r =  gaussian(zCand,mu_mix[i],sig_mix[i])
            p_hat_zCand+=r
        if u<=p_hat_zCand:
            samples.append(zCand)
    return [np.asarray(samples),cnt]


M = get_scale_factor()

samples,trial = rejection_sampler(M,total_samples)
xmax,xmin=[np.ceil(samples.max()),np.floor(samples.min())]
# print(xmax,xmin)
acceptance_rate = total_samples/trial
print("-------acceptance rate is "+str(acceptance_rate)+" ---------")
x_vals = np.linspace(xmax,xmin,300)
total_bins = int(xmax-xmin)
q_x = gaussian(x_vals,mu_p,sig_p)
p_hat_x = 0
for i in range(len(mu_mix)):
    r =  gaussian(x_vals,mu_mix[i],sig_mix[i])
    p_hat_x+=r
plt.xlabel("x")
plt.plot(x_vals,p_hat_x,c='b',label="Unnormalized Distribution")
plt.plot(x_vals,q_x,c='r',label="Proposal Distribution")
plt.hist(samples,total_bins*5,color='g',density = True,label="Histogram")
plt.legend()
plt.show()

plt.plot(x_vals,p_hat_x/3,c='b',label="Normalized Distribution")
plt.plot(x_vals,q_x,c='r',label="Proposal Distribution")
plt.hist(samples,total_bins*5,color='g',density = True,label="Histogram")
plt.legend()

plt.show()
