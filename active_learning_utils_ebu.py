#Implementazione del metodo ebu sfruttando la implementazione standard dell'active learning per Multi-label
import numpy as np


#Se vuoi selezionare solo gli argmin o gli argmax basta commentare ad esempio " highest = sorted_indices[-batch_size:] "
def selections(total_score,top_k_indices ,batch_size,argmin=True):
    selected = []
   
    sorted_indices = np.argsort(total_score)
    lowest = np.array([], dtype=int)
    highest = np.array([], dtype=int)

    if argmin:
        lowest = sorted_indices[:batch_size]        # mancaza di evidenza
    else:
        highest = sorted_indices[-batch_size:]      # I più "conflittuali" (alta evidenza)

    # Mappa gli indici locali (nei top_k) in indici globali del pool
    selected = [top_k_indices[i] for i in np.concatenate([lowest, highest])]


    return selected


# Il procedimento è quello del paper, anche la terminologia usata è congruente
def select_by_ebu_multilabel(X_pool_bin, top_k_indices, batch_size,y_pred,argmin=True):
    print("Ebu started\n")
    

    n_labels = y_pred.shape[1]
    epsilon = 1e-6


    # Estrai solo i candidati selezionati dagli indici
    X_candidates = X_pool_bin[top_k_indices]  


    n_samples = X_candidates.shape[0]
    total_scores = np.zeros(n_samples)


    for label_idx in range(n_labels):
        y_label = y_pred[:, label_idx]
        pos_mask = y_label == 1
        neg_mask = y_label == 0


        # Calcolo delle probabilità condizionate p(x^m | y)
        p_x_given_1 = (X_pool_bin[pos_mask].astype(bool).sum(axis=0) + epsilon) / (pos_mask.sum() + epsilon)
        p_x_given_0 = (X_pool_bin[neg_mask].astype(bool).sum(axis=0) + epsilon) / (neg_mask.sum() + epsilon)


        # Rapporti e maschere per feature informative
        p_ratio = p_x_given_1 / p_x_given_0
        inv_ratio = p_x_given_0 / p_x_given_1


        E1_mask = p_ratio > 1
        E0_mask = inv_ratio > 1


        log_p_ratio = np.zeros_like(p_ratio)
        log_inv_ratio = np.zeros_like(inv_ratio)


        log_p_ratio[E1_mask] = np.log(p_ratio[E1_mask])
        log_inv_ratio[E0_mask] = np.log(inv_ratio[E0_mask])


        log_E1 = np.zeros(n_samples)
        log_E0 = np.zeros(n_samples)


        rows, cols = X_candidates.nonzero()
        for i, j in zip(rows, cols):
            if E1_mask[j]:
                log_E1[i] += log_p_ratio[j]
            if E0_mask[j]:
                log_E0[i] += log_inv_ratio[j]


        total_scores += log_E1 + log_E0
    print("End ebu\n")
    return selections(total_scores,top_k_indices, batch_size,argmin)








'''   PRIMA DEMO
import numpy as np
epsilon = 1e-6
#   F1  F2  F3
m1= [[0, 1, 0],         #matrice dei sample "positivi" nel pool
     [1, 1, 0],
     [0, 1, 1]]
     
m0= [[0, 1, 0],         ##matrice dei sample "negativi" nel pool
     [1, 1, 1],
     [1, 0, 1]]


m1 = np.array(m1)     #trasformo in matrici numpy per applicarci i metodi
m0 = np.array(m0)


#print(m1.astype(bool).sum(axis=1))
print(m1.astype(bool).sum(axis=0) +epsilon )
print( (m1.astype(bool).sum(axis=0) +epsilon ) / (3+epsilon))
print("--------------------------------------------------")
print(m0.astype(bool).sum(axis=0) +epsilon )
print( (m0.astype(bool).sum(axis=0)+epsilon ) / (3+epsilon))


m1 = (m1.astype(bool).sum(axis=0) ) / (3+epsilon)
m0 = (m0.astype(bool).sum(axis=0) ) / (3+epsilon)


print("\nRateo e inv rateo")
p_ratio = m1 / m0         # [0.5 1.5 0.5]  di fatti la feature positiva è solo F2 (>1)
inv_ratio = m0 / m1         # anche non calcolabile
print(p_ratio)
print(inv_ratio)


#mask delle feature     (diviosione in positive e negative)
print("\nMaschere delle feature")
E1_mask = p_ratio > 1
E0_mask = inv_ratio > 1
print(E1_mask)          # [false true false]                
print(E0_mask)          # [duale]    


#Quindi ora presi i sample candidati la sua E1 per questa label sarà la moltiplicazione di solo la F2 se presente
# per il calcolo di E0 invece sarà potenzialmente F1*F3 ma solo se sono presenti e poi calcoli E0*E1


print("\nCalcolo di E1 E0")
E1_vals = np.where(E1_mask, p_ratio, 1.0)  
E0_vals = np.where(E0_mask, inv_ratio, 1.0)
#Se la feature è in Ex_mask, prende il valore p_ratio altrimenti, mette 1.0 (neutro per il prodotto)
print(E1_vals)
print(E0_vals)


print("\ncandidati")
X_candidates = [[0, 1, 0],         #top k uncertain candidati
                [1, 1, 1]]
X_candidates = np.array(X_candidates)
E1 = np.prod(np.where(X_candidates == 1, E1_vals, 1.0), axis=1)
E0 = np.prod(np.where(X_candidates == 1, E0_vals, 1.0), axis=1)
print(E1)          # [1.5 1.5] ovvero la E1 è 1.5 sia per sample[0] che sample[1]
print(E0)          # [1   4  ]   ovvero F1*F3 per entrambi i sample


print("\nTotal scores")
total_scores = np.zeros(len(X_candidates))
total_scores += E1 * E0            #sarà sommato per ogni label
print(total_scores)            # [1.5 6]   Primo sample E1*E0 = 1.5 , secondo 6


print("\n Indici")
idx_min = np.argmin(total_scores)               #quelli a maggior conflitto
idx_max = np.argmax(total_scores)               #quelli a minor conflitto
print(idx_min)
print(idx_max)
'''




'''  DEMO PER LA SELEZIONE
import numpy as np
top_k_indices = [15,25,55,100,150]          #contine l'indice dei top 5 uncertain selezionati
total_scores = np.array([10,22,5,129,1])        #i loro score calcolati


sorted_indices = np.argsort(total_scores)
print(sorted_indices)




lowest = sorted_indices[:2]        #sono i più incerti
print(lowest)


# I più "conflittuali" (alta evidenza)
highest = sorted_indices[-2:]
print(highest)


# Mappa gli indici locali (nei top_k) in indici globali del pool
selected = [top_k_indices[i] for i in np.concatenate([lowest, highest])]
print(selected)
'''




''' Ebu_opt => utilizza matrici sparse
import numpy as np
from scipy.sparse import csr_matrix


# Matrici dense
m1= np.array( [[0, 1, 0],         #matrice dei sample "positivi" nel pool
                [1, 1, 0],
                [0, 1, 1]]
    )
m0= np.array([[0, 1, 0],         ##matrice dei sample "negativi" nel pool
            [1, 1, 1],
            [1, 0, 1]])
epsilon = 1e-6


m1 = csr_matrix(m1)
m0 = csr_matrix(m0)


print("\nMatrice sparsa (CSR) m1:")
print(m1)


print("\nContenuto non-zero m1:")
print(m1.data)
     #".A1" per "flattare"  perchè altrimenti sarebbe [[1 3 1]]
p_x_given_1 = (m1.astype(bool).sum(axis=0).A1 + epsilon) / (3 + epsilon)
p_x_given_0 = (m0.astype(bool).sum(axis=0).A1 + epsilon) / (3 + epsilon)


print("p_x:")
print(p_x_given_1)
print(p_x_given_0)




print("\nRateo e inv rateo")
p_ratio = p_x_given_1 / p_x_given_0         # [0.5 1.5 0.5]  di fatti la feature positiva è solo F2 (>1)
inv_ratio = p_x_given_0 / p_x_given_1         # anche non calcolabile
print(p_ratio)
print(inv_ratio)


print("\nMaschere delle feature")
E1_mask = p_ratio > 1
E0_mask = inv_ratio > 1
print(E1_mask)          # [false true false]                
print(E0_mask)          # [duale]  


log_p_ratio = np.zeros_like(p_ratio)
log_inv_ratio = np.zeros_like(inv_ratio)
print(log_p_ratio)
print(log_inv_ratio)


print("Rateo logaritmico (base 10) filtrato")
log_p_ratio[E1_mask] = np.log(p_ratio[E1_mask])
log_inv_ratio[E0_mask] = np.log(inv_ratio[E0_mask])
print(log_p_ratio)
print(log_inv_ratio)


print("----------------------------------------")
X_candidates = np.array([[0, 1, 0],         #top k uncertain candidati
                        [1, 1, 1]])
X_candidates = csr_matrix(X_candidates)
print(f"Matrice sparsa X_candidates: \n{X_candidates}")


n_samples = X_candidates.shape[0]
total_scores = np.zeros(n_samples)
print(f"Shape di X_candidates (righe): {n_samples}")
print(f"total_scores di partenza: {total_scores}")


log_E1 = np.zeros(n_samples)
log_E0 = np.zeros(n_samples)


rows, cols = X_candidates.nonzero()     #gestione della matrice sparsa
print(f"Righe e colonne {rows},{cols}")      #sono gli indici in cui è presente il valore
for i,j in zip(rows,cols):
    print(f"{i},{j} | ")


for i, j in zip(rows, cols):        #quindi ciclo per ogni valore della matrice (i,j la posizione)
   #j rappresenta la colonna, quindi la "j-esima feature"
    if E1_mask[j]:
        log_E1[i] += log_p_ratio[j]
    if E0_mask[j]:
        log_E0[i] += log_inv_ratio[j]


total_scores += log_E1 + log_E0


print(f"Final total_scores: {total_scores}")
'''

