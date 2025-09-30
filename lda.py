from scipy.special import digamma, polygamma, gammaln
from scipy.special import gamma as gamma_func
import numpy as np

class LDA_Unvectorized:
    def __init__(self, BoW, vocab, num_topics = 5):
        #number of documents
        self.M = BoW.shape[0]

        #number of topics
        self.k = num_topics

        #number of words in vocab
        self.V = BoW.shape[1]
        
        self.BoW = BoW #bag of words
        self.vocab = vocab
        
        rng = np.random.default_rng(seed=0)
        
        self.alpha = np.zeros((self.k)) + 1/self.k + 1e-3 * rng.random(self.k)
        self.alpha = np.maximum(self.alpha, 1e-8)
        
        self.beta = np.zeros((self.k, self.V)) + 1/self.V
        self.beta += 1e-2 * rng.random((self.k, self.V))
        self.beta /= self.beta.sum(axis=1, keepdims=True)

        #get the doc lengths by summing the Bag-of-Words represntations
        self.doc_lengths = self.BoW.sum(axis=1)
        #number of unique words
        self.num_unique_words = (self.BoW!=0).sum(axis=1)

        #phi is not global. It is document specific. \Phi_d has shape (N_d, k), 
        # where N_d is again document specific (number of words in the d-th document)
        #Since different documents might have different lengths, the phi-s will have different shapes so we store in a list
        # since it can't be vectorized
        #for each document, np.zeros((N_d,k)) + 1/k
        #self.phi = [np.zeros((self.doc_lengths[i],self.k)) + 1/self.k for i in range(self.M)]
        self.phi = [np.zeros((self.num_unique_words[i],self.k)) + 1/self.k for i in range(self.M)]

        #gamma shape: [number of documents, number of topics]
        self.gamma = np.broadcast_to(self.alpha, (self.M, self.alpha.size)) + (self.doc_lengths/self.k).reshape(-1, 1)
        
        self.idx_seq = [] #will contain list of unique words (indices), for each document
        self.word_counts_seq = [] #will contain list of appearance counts of each unique word, for each document
        
        self.get_seqs_from_BoW()
            
    def get_seqs_from_BoW(self):
        for b in lda.BoW:
            word_inds = np.where(b!=0)[0]
            self.idx_seq.append(word_inds)
            self.word_counts_seq.append(b[word_inds])
            
        
    def per_document_elbo(self, d):
        #calculate L(\gamma, \phi ; \alpha, \beta)

        #E_q[log p(theta_d | alpha)]
        term1 = (gammaln(np.sum(self.alpha))
                 - np.sum(gammaln(self.alpha))
                 + np.sum((self.alpha-1)*(digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d])))))
        

        #E_q[log p(z_d | theta_d)]
        term2 = np.sum(self.word_counts_seq[d][:, np.newaxis]*self.phi[d]*(digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d]))).reshape(1,-1))

        #E_q[log p(w_d | z_d, beta)]
        term3 = np.sum(self.word_counts_seq[d][:, np.newaxis]*self.phi[d] * np.log(self.beta + 1e-12)[:, self.idx_seq[d]].T)

        #-E_q[log q(theta_d)]
        term4 = - (gammaln(np.sum(self.gamma[d])) 
                   - np.sum(gammaln(self.gamma[d])) 
                   + np.sum((self.gamma[d]-1)*(digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d])))))

        #-E_q[log q(z_d)]
        term5 = - np.sum(self.word_counts_seq[d][:, np.newaxis]*self.phi[d]*np.log(self.phi[d] + 1e-12))

        elbo = term1 + term2 + term3 + term4 + term5
        return elbo
    
    def e_step(self):
        loglikelihood = 0
        for d in range(self.M):
            for r in range(30): #we stop after 30 iterations even if gamma doesn't converge
                gamma_old = self.gamma[d].copy()

                self.phi[d] = (self.beta[:, self.idx_seq[d]] * np.exp(digamma(self.gamma[d])).reshape(-1,1)).T
                self.phi[d] /= self.phi[d].sum(axis=1, keepdims=True) + 1e-12
                self.gamma[d] = self.alpha + (self.word_counts_seq[d][:, np.newaxis]*self.phi[d]).sum(axis=0)
                self.gamma[d] = np.maximum(self.gamma[d], 1e-6)

                #check if gamma converge
                change = np.mean(np.abs(self.gamma[d] - gamma_old))
                if change < 1e-3:
                    break

            loglikelihood += self.per_document_elbo(d) #use this to check if elbo has converged over each main iteration (e-step and m-step)
        return loglikelihood
    
    
    #M-step
    def m_step(self):
        #beta update
        self.beta = np.zeros((self.k, self.V))
        for d in range(self.M):
            self.beta[:, self.idx_seq[d]] += (self.word_counts_seq[d][:, np.newaxis]*self.phi[d]).T
        
        self.beta /= self.beta.sum(axis=1, keepdims=True) + 1e-12


        #alpha update
        for r in range(30): #we stop after 30 iterations even if alpha doesn't converge
            #gradient
            g = (self.M*(digamma(np.sum(self.alpha))-digamma(self.alpha)) 
                 + np.sum(digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True)), axis=0))

            #hessian = diag(h) + 1z1.T
            h = -self.M*polygamma(1, self.alpha)
            z = self.M*polygamma(1, np.sum(self.alpha))
   
            c = np.sum(g/h)/((1/z) + np.sum(1/h))

            h_inv_g = (g-c)/h

            self.alpha -= h_inv_g
            self.alpha = np.maximum(self.alpha, 1e-6)

            if np.linalg.norm(h_inv_g) < 1e-6:
                break
                
    def complete_loop(self, max_iters=100, tol = 1e-3):
        old_loglikelihood = -np.inf
        loglikelihood = 0
        iters = 0
        while iters < max_iters and abs(loglikelihood - old_loglikelihood) > tol:
            old_loglikelihood = loglikelihood
            loglikelihood = self.e_step()
            self.m_step()
            iters += 1
            print(f"EM iter {iters}, ELBO = {loglikelihood:.4f}")

        print(f"Elapsed time (time.time()): {elapsed_time_general:.6f} seconds")

#vectorized version
class LDA:
    def __init__(self, BoW, vocab, num_topics = 5):
        #number of documents
        self.M = BoW.shape[0]

        #number of topics
        self.k = num_topics

        #number of words in vocab
        self.V = BoW.shape[1]
        
        self.BoW = BoW
        self.vocab = vocab
        
        rng = np.random.default_rng(seed=0)
        
        self.alpha = np.zeros((self.k)) + 1/self.k + 1e-3 * rng.random(self.k)
        self.alpha = np.maximum(self.alpha, 1e-8)
        
        self.beta = np.zeros((self.k, self.V)) + 1/self.V
        self.beta += 1e-2 * rng.random((self.k, self.V))
        self.beta /= self.beta.sum(axis=1, keepdims=True)

        #get the doc lengths by summing the Bag-of-Words represntations
        self.doc_lengths = self.BoW.sum(axis=1)
        #number of unique words
        self.num_unique_words = (self.BoW!=0).sum(axis=1)
        
        #gamma shape: [number of documents, number of topics]
        self.gamma = np.broadcast_to(self.alpha, (self.M, self.alpha.size)) + (self.doc_lengths/self.k).reshape(-1, 1)

        #docs_list -> non-zero entry row index list
        #words_list -> non-zero entry column index list
        #counts_data -> non-zero entry value list
        self.docs_list, self.words_list = np.where(self.BoW != 0)
        self.counts_data = self.BoW[self.docs_list, self.words_list]
        
        self.phi = np.zeros((self.counts_data.shape[0], self.k)) + 1/self.k

    def get_elbo(self):
        #calculate L(\gamma, \phi ; \alpha, \beta)

        #E_q[log p(theta | alpha)]
        term1 = (gammaln(np.sum(self.alpha))*self.M
                 - np.sum(gammaln(self.alpha))*self.M
                 + np.sum((self.alpha-1)*(digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True)))))
        

        #E_q[log p(z | theta)]
        gamma_part = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True))
        term2 = np.sum(self.counts_data[:, np.newaxis]*self.phi*gamma_part[self.docs_list, :])

        
        #E_q[log p(w | z, beta)]
        term3 = np.sum(self.counts_data[:, np.newaxis]*self.phi*np.log(self.beta + 1e-12)[:, self.words_list].T)

        #-E_q[log q(theta)]
        term4 = - np.sum(gammaln(np.sum(self.gamma, axis=1, keepdims=True)) 
                   - np.sum(gammaln(self.gamma), axis=1, keepdims=True) 
                   + np.sum((self.gamma-1)*(digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True))), axis=1, keepdims=True))


        #-E_q[log q(z)]
        term5 = - np.sum(self.counts_data[:, np.newaxis]*self.phi*np.log(self.phi + 1e-12))

        elbo = term1 + term2 + term3 + term4 + term5
        return elbo
    
    def e_step(self):
        for r in range(30):
            self.phi = self.beta[:, self.words_list].T * np.exp(digamma(self.gamma))[self.docs_list]
            self.phi /= self.phi.sum(axis=1, keepdims=True) + 1e-12
            
            step1 = self.counts_data[:, np.newaxis]*self.phi
            self.gamma = self.alpha + np.vstack([np.bincount(self.docs_list, weights=step1[:,i]) for i in range(self.k)]).T
            self.gamma = np.maximum(self.gamma, 1e-6)
        return self.get_elbo()
    
    #M-step
    def m_step(self):
        #beta update
        step1 = self.counts_data[:, np.newaxis]*self.phi
        self.beta = np.vstack([np.bincount(self.words_list, weights=step1[:,i]) for i in range(self.k)])
        
        self.beta /= self.beta.sum(axis=1, keepdims=True) + 1e-12


        #alpha update
        for r in range(30): #we stop after 30 iterations even if alpha doesn't converge
            #gradient
            g = (self.M*(digamma(np.sum(self.alpha))-digamma(self.alpha)) 
                 + np.sum(digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True)), axis=0))

            #hessian = diag(h) + 1z1.T
            h = -self.M*polygamma(1, self.alpha)
            z = self.M*polygamma(1, np.sum(self.alpha))
   
            c = np.sum(g/h)/((1/z) + np.sum(1/h))

            h_inv_g = (g-c)/h

            self.alpha -= h_inv_g
            self.alpha = np.maximum(self.alpha, 1e-6)

            if np.linalg.norm(h_inv_g) < 1e-6:
                break
                
    def complete_loop(self, max_iters=100, tol = 1e-3):
        old_loglikelihood = -np.inf
        loglikelihood = 0
        iters = 0
        while iters < max_iters and abs(loglikelihood - old_loglikelihood) > tol:
            old_loglikelihood = loglikelihood
            loglikelihood = self.e_step()
            self.m_step()
            iters += 1
            print(f"EM iter {iters}, ELBO = {loglikelihood:.4f}")
        
