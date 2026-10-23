import numpy as np

def blahut_arimoto(W, max_iter=1000, tol=1e-6):
    """
    Calcule la capacité d'un canal et la distribution optimale via Blahut-Arimoto.
    
    W : tableau numpy de forme (nX, nY)
        Matrice de transition W[i, j] = P(Y=y_j | X=x_i).
        Chaque ligne doit sommer à 1.
        
    Retourne :
        C : Capacité en bits.
        p_x : Distribution de probabilité optimale sur l'entrée X.
    """
    nX, nY = W.shape
    
    # Normalisation par sécurité (somme sur y doit valoir 1)
    W = W / np.sum(W, axis=1, keepdims=True)
    
    # Initialisation uniforme
    p_x = np.ones(nX) / nX
    
    for _ in range(max_iter):
        p_y = p_x @ W
        
        # Q(x|y) = p(x) * W(y|x) / p(y)
        # On utilise une petite constante pour éviter les divisions par zéro
        Q = (p_x[:, None] * W) / (p_y[None, :] + 1e-12)
        
        # Nouvelle p(x) = exp( sum_y W(y|x) log Q(x|y) )
        log_p_x = np.sum(W * np.log(Q + 1e-12), axis=1)
        log_p_x -= np.max(log_p_x) # Stabilité numérique
        p_x_new = np.exp(log_p_x)
        p_x_new /= np.sum(p_x_new)
        
        if np.max(np.abs(p_x_new - p_x)) < tol:
            p_x = p_x_new
            break
            
        p_x = p_x_new
        
    # Calcul de l'Information Mutuelle finale
    p_y = p_x @ W
    C = 0.0
    for i in range(nX):
        if p_x[i] > 1e-12:
            for j in range(nY):
                if W[i, j] > 1e-12 and p_y[j] > 1e-12:
                    C += p_x[i] * W[i, j] * np.log2(W[i, j] / p_y[j])
                    
    return C, p_x

def mutual_information_uniform(W):
    """
    Calcule l'Information Mutuelle pour une distribution d'entrée *uniforme*.
    Utile pour comparer si on n'optimise pas les probabilités.
    """
    nX, nY = W.shape
    W = W / np.sum(W, axis=1, keepdims=True)
    p_x = np.ones(nX) / nX
    p_y = p_x @ W
    
    C = 0.0
    for i in range(nX):
        for j in range(nY):
            if W[i, j] > 1e-12 and p_y[j] > 1e-12:
                C += p_x[i] * W[i, j] * np.log2(W[i, j] / p_y[j])
    return C
