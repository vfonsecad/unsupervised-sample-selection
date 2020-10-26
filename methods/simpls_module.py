# ------------------------------------------------------------------------

#                               simpls functions with numba
# by: valeria fonseca diaz
# supervisors: Wouter Saeys, Bart De Ketelaere
# ------------------------------------------------------------------------


import numpy as np
import numba


@numba.njit(fastmath=True)
def simpls_loadings(xx, yy, P, ncp):
    
    xx = np.ascontiguousarray(xx)
    yy = np.ascontiguousarray(yy)
    P = np.ascontiguousarray(P)
        
    mu_x = ((P.dot(xx)).sum(axis=0))/ P.sum()
    mu_y = ((P.dot(yy)).sum(axis=0))/ P.sum()
        

    Xc = xx.copy() - mu_x
    Yc = yy.copy() - mu_y

    N = Xc.shape[0]
    K = Xc.shape[1]
    q = Yc.shape[1]

    R = np.zeros((K, ncp))  # Weights to get T components
    V = np.zeros((K, ncp))  # orthonormal base for X loadings
    S = Xc.T.dot(P).dot(Yc)  # cov matrix
    r = np.zeros((K, 1))
    v = np.zeros((K, 1))

    aa = 0

    while aa < ncp:
        
        r[:,0] = S[:,0].flatten()
        
#         if q > 1:
                
#             U0, sval, Qsvd = np.linalg.svd(S)
#             Sval = np.zeros((U0.shape[0], Qsvd.shape[0]))
#             Sval[0:sval.shape[0], 0:sval.shape[0]] = np.diag(sval)

#             r[:,0] = U0.dot(Sval)[:,0]

                
        tt = Xc.dot(r)
        tt = tt - ((P.dot(tt)).sum(axis=0)/ P.sum())
        TT_scale = np.sqrt(tt.T.dot(P).dot(tt))
            # Normalize
        tt = tt / TT_scale            
        r = r / TT_scale
        
        p = Xc.T.dot(P).dot(tt)
        v[:,0] = p.flatten()
       
        
        if aa > 0:
            
            v = v - V.dot(V.T.dot(p))

        v = v / np.sqrt(v.T.dot(v))
        S = S - v.dot(v.T.dot(S))

        R[:, aa] = r.flatten()
        V[:, aa] = v.flatten()
        
        aa += 1

    return R
    

@numba.njit(fastmath=True)
def simpls_fit(xx, yy, ncp):
    
    
    X = np.ascontiguousarray(xx)
    Y = np.ascontiguousarray(yy)
    
    N = X.shape[0]
    K = X.shape[1]
    q = Y.shape[1]
    
    mu_x = np.zeros((1,K))
    mu_y = np.zeros((1,q))
    
    BPLS = np.zeros((K, ncp,q))  # final regression vector for every lv
    
 
    P = np.identity(N) # kept so that maybe weights for samples can we changed
    
    mu_x[0,:] = ((P.dot(X)).sum(axis=0)) / P.sum()
    Xc = X - mu_x
    mu_y[0,:] = ((P.dot(Y)).sum(axis=0)) / P.sum()
    Yc = Y - mu_y


    R = simpls_loadings(X, Y, P, ncp)
           
    
               
    # --- simpls final regression
    
    
    aa = 0
    
    y_mu = np.zeros((ncp,q))

    while aa < ncp:
        
        
        current_R = np.ascontiguousarray(R[:,0:(aa+1)])
        TT = Xc.dot(current_R)
        
        tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), TT), axis=1)
        wtemp = np.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))
            
        wtemp_bool = np.zeros(aa+2, dtype=np.int64) == 0  
        wtemp_bool[0] = False
    
        wtemp1 = wtemp[wtemp_bool, :]        
       
        BPLS_aa = np.dot(current_R,wtemp1)
        
        BPLS[:,aa,:] = BPLS_aa
        y_mu[aa,:] = wtemp[0:1, :]
        
        aa += 1
 
    x_mu = mu_x.copy()
    
    return (BPLS,x_mu, y_mu,P)



@numba.njit(fastmath=True)
def simpls_predict(X, BPLS, x_mu, y_mu):
    
    Ypred = np.zeros((X.shape[0], BPLS.shape[1],BPLS.shape[2]))
    
    for aa in range(BPLS.shape[1]):
        
        current_BPLS = np.ascontiguousarray(BPLS[:,aa,:])
        
        Ypred[:,aa,:] = y_mu[aa,:] + (X - x_mu).dot(current_BPLS)

    return Ypred

@numba.njit(fastmath=True)
def rmse(yy, y_pred, sample_weights):
    
    total_ncp = y_pred.shape[1]
    rmse = np.zeros((total_ncp,y_pred.shape[2]))
    
    N = yy.shape[0]
    P = np.diag(sample_weights.flatten()) / sample_weights.flatten().sum()
    
    for aa in range(total_ncp):   
        
        r = yy.copy() - y_pred[:,aa,:]
        r = np.power(r,2)
        msep = ((P.dot(r)).sum(axis=0)) / P.sum()
        rmse[aa,:] = np.sqrt(msep)

    return rmse  

@numba.njit(fastmath=True)
def r2(yy, y_pred, sample_weights):
    
    total_ncp = y_pred.shape[1]
    r2 = np.zeros((total_ncp,y_pred.shape[2]))
    
    N = yy.shape[0]
    P = np.diag(sample_weights.flatten()) / sample_weights.flatten().sum()
    yy_mu = ((P.dot(yy)).sum(axis=0)) / P.sum()
    
    for aa in range(total_ncp):   
        
        r = yy.copy() - y_pred[:,aa,:]
        sq_r = np.power(r,2)
        numerator = ((P.dot(sq_r)).sum(axis=0))
        y_diff = yy.copy() - yy_mu
        sq_y = np.power(y_diff,2)
        denominator = ((P.dot(sq_y)).sum(axis=0))
        r2[aa,:] = 1-numerator/denominator

    return r2 

    


@numba.njit(fastmath=True)
def simpls_univariate_cv(xx, yy, total_ncp, number_splits=10):
    
    
    X = xx.copy()
    Y = yy.copy()

    N = X.shape[0]
    K = X.shape[1]
    q = Y.shape[1]
    
    size_split = int(N/number_splits)
    sample_in_group = np.arange(0,number_splits)
    
    for ss in range(size_split):
        sample_in_group = np.concatenate((sample_in_group, np.arange(0,number_splits)), axis = 0)
        
    sample_in_group_shuffled = np.random.permutation(sample_in_group[0:N])
        

    cv_predicted = np.zeros((N,total_ncp,q))
    
    sample_weights = np.ones(N)
            
        
    for ss in range(number_splits):
        
      
                
                
        test_obs = np.zeros(N, dtype=np.int64) == 1  
        cal_obs = np.zeros(N, dtype=np.int64) == 0 
        test_obs[np.where(sample_in_group_shuffled == ss)[0]] = True
        cal_obs[np.where(sample_in_group_shuffled == ss)[0]] = False                
        
        trained = simpls_fit(X[cal_obs,:], Y[cal_obs,:], ncp=total_ncp)
        predicted = simpls_predict(X[test_obs,:], trained[0], trained[1], trained[2])                
        cv_predicted[test_obs,:,:] = predicted

            
    rmsecv = rmse(Y, cv_predicted, sample_weights = sample_weights)[:,0]
    r2cv = r2(Y, cv_predicted, sample_weights = sample_weights)[:,0]


        
    return (rmsecv,r2cv)
    