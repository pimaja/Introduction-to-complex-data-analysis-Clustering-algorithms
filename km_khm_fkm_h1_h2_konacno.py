import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import choice
from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as LA


def KM(x,c):
	suma = 0

	for i in range(np.size(x[:,0])):
		min_norma = np.power( LA.norm(x[i]-c[0]) , 2 )

		for j in range(1, np.size(c[:,0])):
			pom = np.power( LA.norm(x[i]-c[j]) , 2 )
			if(pom < min_norma):
				min_norma = pom

		suma = suma + min_norma

	return suma


def m_km(c,cj,xi):
	min_norma = np.power( LA.norm(xi-c[0]) , 2 )
	min_centar = np.array(c[0])

	for s in range(1,np.size(c[:,0])):
		pom = np.power( LA.norm(xi-c[s]) , 2 )
		if(pom < min_norma):
			min_norma = pom
			min_centar = c[s]

	if( np.array_equal( min_centar, cj ) ):
		return 1

	else:
		return 0


def w_km(xi):
	return 1


def re_cj_km(x,c,cj):
	brojnik1 = 0
	brojnik2 = 0
	nazivnik = 0

	for i in range( np.size(x[:,0]) ):
		t = m_km(c,cj,x[i]) * w_km(x[i])

		brojnik1 += t * x[i,0]
		brojnik2 += t * x[i,1]	
		nazivnik += t

	if(nazivnik == 0):
		novi_centar = np.array([0,0])
		novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,1]) )
		novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
		return novi_centar

	novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
	return novi_centar


def m_fkm(c, cj, xi):
    r = 1.5
    e = 0.00000001
    
    nazivnik = 0
    for j in range(1, np.size(c[:,0])):
           nazivnik += np.power( np.max([LA.norm(xi-c[j]), e]), -2/(r-1) )    
    brojnik = np.power( np.max([LA.norm(xi-cj), e]), -2/(r-1) )
    return brojnik/nazivnik


def w_fkm(xi):
    return 1


def re_cj_fkm(x,c,cj):
    brojnik1 = 0
    brojnik2 = 0
    nazivnik = 0

    for i in range( np.size(x[:,0]) ):
        t = m_fkm(c,cj,x[i]) * w_fkm(x[i])
                                                
        brojnik1 += t * x[i,0]
        brojnik2 += t * x[i,1]
        nazivnik += t

    if(nazivnik == 0):
        novi_centar = np.array([0,0])
        novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
        novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
        return novi_centar

    novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
    return novi_centar  


def KHM(x,c):
	p = 3.5
	e = 0.00000001

	suma = 0

	for i in range( np.size(x[:,0]) ):
		nazivnik = 0

		for j in range(np.size(c[:,0])):
			if( np.array_equal( x[i], cj ) ):
				pom = np.power(e,p)

			else:
				pom = np.power( LA.norm(x[i]-c[j]) , p )

			pomocni = 1 / pom
			nazivnik += pomocni

		suma += (np.size(c[:,0]) / nazivnik)

	return suma


def m_khm(c,cj,xi):
	p = 3.5
	e = 0.00000001

	brojnik = np.power( np.max([LA.norm(xi-cj), e]), -p-2 )
	nazivnik = 0

	for j in range( np.size(c[:,0]) ):
		pom_nazivnik = np.power( np.max([LA.norm(xi-c[j]), e]), -p-2 )
		nazivnik += pom_nazivnik

	rez = brojnik / nazivnik
	return rez


def w_khm(c,xi):
	p = 3.5
	e = 0.00000001

	brojnik = 0
	nazivnik = 0

	for j in range( np.size(c[:,0]) ):
		pom_brojnik = np.power( np.max([LA.norm(xi-c[j]), e]) , -p-2)
		brojnik += pom_brojnik

		pom_nazivnik = np.power( np.max([LA.norm(xi-c[j]), e]) , -p )
		nazivnik += pom_nazivnik

	nazivnik = np.power(nazivnik , 2)

	rez = brojnik / nazivnik
	return rez


def re_cj_khm(x,c,cj):
	brojnik1 = 0
	brojnik2 = 0
	nazivnik = 0

	for i in range( np.size(x[:,0]) ):
		t = m_khm(c,cj,x[i]) * w_khm(c,x[i])
												
		brojnik1 = brojnik1 + t * x[i,0]
		brojnik2 = brojnik2 + t * x[i,1]										
		nazivnik = nazivnik + t

	if(nazivnik == 0):
		novi_centar = np.array([0,0])
		novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
		novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
		return novi_centar

	novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
	return novi_centar


def m_h1(c,cj,xi):  ##isto kao i m_km
    min_norma = np.power( LA.norm(xi-c[0]) , 2 )
    min_centar = np.array(c[0])

    for s in range(1,np.size(c[:,0])):
        pom = np.power( LA.norm(xi-c[s]) , 2 )
        if(pom < min_norma):
            min_norma = pom
            min_centar = c[s]

    if( np.array_equal( min_centar, cj ) ):
        return 1

    else:
        return 0
    

def w_h1(c,xi):  ##isto kao i w_khm
    p = 3.5
    e = 0.00000001

    brojnik = 0
    nazivnik = 0

    for j in range( np.size(c[:,0]) ):
        pom_brojnik = np.power( np.max([LA.norm(xi-c[j]), e]) , -p-2)
        brojnik += pom_brojnik

        pom_nazivnik = np.power( np.max([LA.norm(xi-c[j]), e]) , -p )
        nazivnik += pom_nazivnik

    nazivnik = np.power(nazivnik , 2)

    rez = brojnik / nazivnik
    return rez


def re_cj_h1(x,c,cj):
    brojnik1 = 0
    brojnik2 = 0
    nazivnik = 0

    for i in range( np.size(x[:,0]) ):
        t = m_h1(c,cj,x[i]) * w_h1(c,x[i])
                                                
        brojnik1 = brojnik1 + t * x[i,0]
        brojnik2 = brojnik2 + t * x[i,1]
        nazivnik = nazivnik + t

    if(nazivnik == 0):
        novi_centar = np.array([0,0])
        novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
        novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
        return novi_centar

    novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
    return novi_centar

def m_h2(c,cj,xi): ## isto kao i m_khm
    p = 3.5
    e = 0.00000001

    brojnik = np.power( np.max([LA.norm(xi-cj), e]), -p-2 )
    nazivnik = 0

    for j in range( np.size(c[:,0]) ):
        pom_nazivnik = np.power( np.max([LA.norm(xi-c[j]), e]), -p-2 )
        nazivnik += pom_nazivnik

    rez = brojnik / nazivnik
    return rez


def w_h2(xi): ## isto kao i w_km
    return 1


def re_cj_h2(x,c,cj):
    brojnik1 = 0
    brojnik2 = 0
    nazivnik = 0

    for i in range( np.size(x[:,0]) ):
        t = m_h2(c,cj,x[i]) * w_h2(x[i])
                                                
        brojnik1 = brojnik1 + t * x[i,0]
        brojnik2 = brojnik2 + t * x[i,1]
        nazivnik = nazivnik + t

    if(nazivnik == 0):
        novi_centar = np.array([0,0])
        novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
        novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
        return novi_centar

    novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
    return novi_centar


def re_cj_triem(x,c,cj):
	brojnik1 = 0
	brojnik2 = 0
	nazivnik = 0

	for i in range( np.size(x[:,0]) ):
		t = m_fkm(c,cj,x[i]) * w_khm(c,x[i])
                                                
		brojnik1 = brojnik1 + t * x[i,0]
		brojnik2 = brojnik2 + t * x[i,1]
		nazivnik = nazivnik + t

	if(nazivnik == 0):
		novi_centar = np.array([0,0])
		novi_centar[0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
		novi_centar[1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
		return novi_centar

	novi_centar = np.array( [brojnik1/nazivnik , brojnik2/nazivnik] )
	return novi_centar


def normalize(x):
	x[:,0] = ( x[:,0] - np.mean(x[:,0]) ) / np.std(x[:,0])
	x[:,1] = ( x[:,1] - np.mean(x[:,1]) ) / np.std(x[:,1])

	return x


def belong(x,c,k):
	prip = np.zeros( np.size(x[:,0]) )

	for i in range( np.size(x[:,0]) ):
		min_norma = LA.norm(x[i]-c[0])

		for j in range(1,k):
			pom_norma = LA.norm(x[i]-c[j])
			if(pom_norma < min_norma):
				min_norma = pom_norma
				prip[i] = j

	return prip


def graph(x,c,s1,s2,s3):
	b = belong(x,c,k)

	plt.figure()
	plt.scatter(x[:,0], x[:,1], c=b, s=25, cmap="plasma", edgecolors='black')
	plt.scatter(c[:,0], c[:,1], marker='o', c='black', s=40)
	plt.suptitle(s1, fontsize=14, color='navy', y=0.04)
	cost_f = "Cost function: " + str(round(s2, 6))
	plt.figtext(0.12, 0.9, cost_f, fontsize=10, color='black')
	score_c = "Score of clustering: " + str(round(s3, 6))
	plt.figtext(0.57, 0.9, score_c, fontsize=10, color='black')
	plt.savefig(s1 + ".png")


def optimal_centers(x,y,k):
	opt_centri = np.zeros((k,2))
	kolicina = np.zeros((k,2))
	
	for i in range( np.size(x[:,0]) ):
		j = y[i]
		opt_centri[j] += x[i]
		kolicina[j] += [1,1]

	for j in range(k):
		if(kolicina[j,0] == 0):
			kolicina[j] = [1,1]
			opt_centri[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
			opt_centri[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
	
	opt_centri = opt_centri / kolicina
	return opt_centri


def Forgy_initialization(x,k):	
    indeksi = choice( np.size(x[:,0]), k,replace=False)		#replace znaci ponavljanje

    centri_km = []
    centri_khm = []
    centri_fkm = []
    centri_h1 = []
    centri_h2 = []

    for j in range(k):
        centri_km.append( x[indeksi[j]] )
        centri_khm.append( x[indeksi[j]] )
        centri_fkm.append( x[indeksi[j]] )
        centri_h1.append( x[indeksi[j]] )
        centri_h2.append( x[indeksi[j]] )

    centri_km = np.array(centri_km)
    centri_khm = np.array(centri_khm)
    centri_fkm = np.array(centri_km)
    centri_h1 = np.array(centri_km)
    centri_h2 = np.array(centri_km)

    return centri_km, centri_khm, centri_fkm, centri_h1, centri_h2


def RP_initialization(x,k):
    rand_part = choice(k, np.size(x[:,0]) )

    centri_km = np.zeros((k,2))
    centri_khm = np.zeros((k,2))
    centri_fkm = np.zeros((k,2))
    centri_h1 = np.zeros((k,2))
    centri_h2 = np.zeros((k,2)) 
    kolicine = np.zeros((k,2))

    for i in range( np.size(x[:,0]) ):
        j = rand_part[i]
        centri_km[j] += x[i]
        centri_khm[j] += x[i]
        centri_fkm[j] += x[i]
        centri_h1[j] += x[i]
        centri_h2[j] += x[i]
        kolicine[j] += [1,1]

    for j in range(k):
        if(kolicine[j,0] == 0):
            kolicine[j] = [1,1]
            centri_km[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
            centri_km[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
            centri_khm[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
            centri_khm[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
            centri_fkm[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
            centri_fkm[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
            centri_h1[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
            centri_h1[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )
            centri_h2[j,0] = np.random.uniform( np.min(x[:,0]), np.max(x[:,0]) )
            centri_h2[j,1] = np.random.uniform( np.min(x[:,1]), np.max(x[:,1]) )

    centri_km = centri_km / kolicine
    centri_khm = centri_khm / kolicine
    centri_fkm = centri_fkm / kolicine
    centri_h1 = centri_h1/ kolicine
    centri_h2 = centri_h2/ kolicine

    return centri_km, centri_khm, centri_fkm, centri_h1, centri_h2


def start(X,C,f):
    brojac = 0
    while(True):
        stari_centri = C

        for j in range(k):
            C[j] = f(X,C,C[j])

        if( np.array_equal(stari_centri, C) ):
            brojac += 1
            if brojac==10: break
    return C


def result(algorithm,initialization,C,optimalniC,X):
    print("\n" + algorithm + " - " + initialization + " quality: %.16f" %math.sqrt( KM(X,C) ))
    rd = math.sqrt( KM(X,C) / KM(X,optimalniC) )
    print("Score of a clustering: %.16f" %rd)
    graph(X, C, algorithm + " i " + initialization + " inicijalizacija", math.sqrt( KM(X,C) ), rd)
    plt.show()


if __name__ == '__main__':
    n = 500
    k = 10

    x,y = make_blobs(n_samples=n,n_features=2,centers=k,cluster_std=0.8)
    x = normalize(x)

    optimalni_centri = optimal_centers(x,y,k)
    rd = math.sqrt( KM(x,optimalni_centri) / KM(x,optimalni_centri) )
    graph(x,optimalni_centri,"Originalni podaci i optimalni centri", math.sqrt( KM(x,optimalni_centri)), rd)
    print("\nOptimal quality: %.16f" %math.sqrt( KM(x,optimalni_centri) ))

    forgy_centri_km, forgy_centri_khm, forgy_centri_fkm, forgy_centri_h1, forgy_centri_h2 = Forgy_initialization(x,k)
    rd = math.sqrt( KM(x,forgy_centri_km) / KM(x,optimalni_centri) )
    graph(x, forgy_centri_km,"Originalni podaci i Forgy inicijalizacija", math.sqrt( KM(x,forgy_centri_km)), rd)

    rp_centri_km, rp_centri_khm, rp_centri_fkm, rp_centri_h1, rp_centri_h2 = RP_initialization(x,k)
    rd = math.sqrt( KM(x,rp_centri_km) / KM(x,optimalni_centri) )
    graph(x, rp_centri_km, "Originalni podaci i RP inicijalizacija", math.sqrt( KM(x,rp_centri_km)), rd)

    plt.show()


    #### KM - FORGY
    forgy_centri_km = start(x,forgy_centri_km,re_cj_km)
    rd = math.sqrt( KM(x,forgy_centri_km) / KM(x,optimalni_centri) )
    result("KM", "Forgy", forgy_centri_km, optimalni_centri, x)
        
    #### KM - RANDOM PARTITION
    rp_centri_km = start(x,rp_centri_km,re_cj_km)    
    result("KM", "RP", rp_centri_km, optimalni_centri, x)
    
    #### FKM - FORGY
    forgy_centri_fkm = start(x,forgy_centri_fkm,re_cj_fkm)
    result("FKM", "Forgy", forgy_centri_fkm, optimalni_centri, x)

    #### FKM - RANDOM PARTITION
    rp_centri_fkm = start(x,rp_centri_fkm,re_cj_fkm)
    result("FKM", "RP", rp_centri_fkm, optimalni_centri, x)

    #### KHM - FORGY
    forgy_centri_khm = start(x,forgy_centri_khm,re_cj_khm)
    result("KHM", "Forgy", forgy_centri_khm, optimalni_centri, x)

    #### KHM - RANDOM PARTITION
    rp_centri_khm = start(x,rp_centri_khm,re_cj_khm)    
    result("KHM", "RP", rp_centri_khm, optimalni_centri, x)
    
    #### H1 - FORGY
    forgy_centri_h1 = start(x,forgy_centri_h1,re_cj_h1)
    result("H1", "Forgy", forgy_centri_h1, optimalni_centri, x)

    #### H1 - RANDOM PARTITION
    rp_centri_h1 = start(x,rp_centri_h1,re_cj_h1)    
    result("H1", "RP", rp_centri_h1, optimalni_centri, x)
    
    #### H2 - FORGY
    forgy_centri_h2 = start(x,forgy_centri_h2,re_cj_h2)
    result("H2", "Forgy", forgy_centri_h2, optimalni_centri, x)

    #### H2 - RANDOM PARTITION
    forgy_centri_triem = rp_centri_h2
    rp_centri_triem = rp_centri_h2
    rp_centri_h2 = start(x,rp_centri_h2,re_cj_h2)    
    result("H2", "RP", rp_centri_h2, optimalni_centri, x)


    ### Triem - FORGY
    forgy_centri_triem = start(x,forgy_centri_triem,re_cj_triem)
    result("TrieM", "Forgy", forgy_centri_triem, optimalni_centri, x)

    ### Triem - RP
    rp_centri_triem = start(x,rp_centri_triem,re_cj_triem)
    result("TrieM", "RP", rp_centri_triem, optimalni_centri, x)