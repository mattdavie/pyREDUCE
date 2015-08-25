def flat_sp_func(order,ycen,osample=10,lamb_sp=0.0,lambda_sf=0.1,use_mask=0,noise=5.85,uncert=False,im_output=False,normflat=False,slitfunc=False):
	#order:		order as spatial pixel by dispersion pixel array
	#ycen:		coordinates of the centre along the order in the dispersion direction
	#osample:	oversampling rate
	#lamb_sp:	spectral step
	#lamb_sf:	slitfunction step
	#use_mask:	mask - same shape as order, 1 where pixel is good, else 0. If not supplied one will be constructed (crudely)
	#noise: 	noise value with default 5.85
	#uncert:	array of uncertainties if passing them into this method
	#im_output:	do we wish to output a reconstruction at the end?
	#normflat:	are we passing in a normalised flat field for this?
	#slitfunc:	do we wish to output the slitfunctions at the end?
	
	#Obtain dimensions of the input order to be reduced
	nrow,ncol=order.shape
	n=(nrow+1)*osample+1

	#Mask creation
	try:
		if (use_mask==0):
			mask=np.ones((nrow, ncol))
			mask[np.where(order > 60000)]=0.0
	except:
		mask=np.copy(use_mask)

	y=np.arange(n)/float(osample)-1.
	bklind=np.arange(osample+1)+n*osample
	oind=np.arange(osample+1)*(osample+2)
	olind=oind[0:osample+1]
	for m in range(osample+1, 2*osample+1):
		mm=m-osample
		bklind=np.append(bklind, np.arange(osample+1-mm)+n*m)
		olind=np.append(olind, oind[0:osample-mm+1]+mm)

	#Now create spectrum
	sp=np.repeat(0,ncol)
	sf=np.sum(order*mask,axis=1)

	#Construction of the first slit function and spectrum
	#Counting median widths and
	#Estimating the first part of the slit function
	
	sf_med=np.arange(sf.shape[0]-4)
	for i in range(2, sf.shape[0]-2):
		sf_med[i-2]=np.median(sf[i-2:i+3])
	sf[2:sf.shape[0]-2]=sf_med
	sf=sf/np.sum(sf)
	
	#We estimate the first bit of the spectrum
	
	sp=np.sum((order*mask)*(np.outer(sf,np.repeat(1,ncol))),axis=0)
	sp_med=np.arange(sp.shape[0]-4)
	for i in range(2, sp.shape[0]-2):
		sp_med[i-2]=np.median(sp[i-2:i+3])
	sp[2:sp.shape[0]-2]=sp_med
	sp=sp/np.sum(sp)*np.sum(order*mask)
	dev = np.sqrt(np.sum(mask*(order-np.outer(sf, sp))**2)/np.sum(mask))
	mask[np.where(abs(order-np.outer(sf, sp)) > 3.*dev)]=0.0

	#Calculate weights
	weight=1./np.float64(osample)

########################################################################
	for iter in range(1,25):
	#build a matrix with Omega as the diagonal
		Akl=np.zeros((2*osample+1,n))
		Bl=np.zeros((1,n))
		omega=np.repeat(weight,osample+1)
		for i in range(0, ncol):
			#Creating arrays by weight
			omega=np.repeat(weight,osample+1)
			yy=y+ycen[i]
			ind=np.where((yy>=0.0) & (yy<1.))[0]
			i1=ind[0]
			i2=ind[-1]
			omega[0]=yy[i1]
			omega[-1]=1.-yy[i2]
			o=np.outer(omega,omega)
			o[osample,osample]=o[osample,osample]+o[0,0]   
			bkl=np.zeros((2*osample+1,n))
			omega_t=np.reshape(o, o.shape[0]*o.shape[1])
			oo= omega_t[olind]
			for l in range(0, nrow):
				bkl_temp=np.reshape(bkl, bkl.shape[0]*bkl.shape[1])
				t=l*osample+bklind+i1
				bkl_temp[t]=oo*mask[l,i]
			bkl=np.reshape(bkl_temp,(2*osample+1,n))
			oo=o[osample, osample]
			for l in range(1, nrow):
				bkl[osample,l*osample+i1]=oo*mask[l,i]
			bkl[osample,nrow*osample+i1]=omega[osample]**2*mask[nrow-1,i]
			for m in range (0,osample):
				bkl[m,(osample-m):(n)]=bkl[2*osample-m,0:(n-osample+m)]
			Akl=Akl+(sp[i]**2)*bkl
			o=np.zeros((1,n))
			for l in range (0, nrow):
				o[0,l*osample+i1:l*osample+i1+osample+1]=order[l,i]*weight*mask[l,i]
			for l in range (1, nrow):
				o[0,l*osample+i1]=order[l-1,i]*omega[osample]*mask[l-1,i]+order[l,i]*omega[0]*mask[l,i]
			o[0,i1]=order[0,i]*omega[0]*mask[0,i]
			o[0,nrow*osample+i1]=order[nrow-1,i]*omega[osample]*mask[nrow-1,i]
			Bl=Bl+sp[i]*o

		tab=np.zeros((n,2))
		lamda=lambda_sf*np.sum(Akl[osample,:])/n
		lambda_tab=np.zeros((1,n))
		for elem in range(0,n):
			lambda_tab[0,elem]=lamda
	
		Akl[osample,0]=Akl[osample,0]+lambda_tab[0,0]
		Akl[osample,n-1]=Akl[osample,n-1]+lambda_tab[0,n-1]
		Akl[osample,1:n-1]=Akl[osample,1:n-1]+2.*lambda_tab[0,1:n-1]
		Akl[osample+1,0:n-1]=Akl[osample+1,0:n-1]-lambda_tab[0,0:n-1]
		Akl[osample-1,1:n]=Akl[osample-1,1:n]-lambda_tab[0,1:n]
		Bl=Bl.T
		
		x = solve_banded((osample,osample), Akl, Bl, overwrite_ab=True, overwrite_b=True)
		ind0=[np.where(x<0)]
		x[ind0]=0.0
		sf=x/np.sum(x)*osample
		r=np.repeat(0.,sp.shape[0])
		sp_old=np.copy(sp)
		dev_new=0.0
		
		for i in range(0, ncol):
			omega=np.repeat(weight,osample)
			yy=y+ycen[i]
			ind1=np.where((yy>=0.0) & (yy<nrow))[0]
			i1=ind1[0]
			i2=ind1[-1]
			omega[0]=yy[i1]
			ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
			o=np.dot(ssf,omega)

			yyy=nrow-yy[i2]
			o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
			o[nrow-1]=o[nrow-1]+sf[i2]*yyy
			r[i]=np.dot((order[:,i]*mask[:,i]),o)
			sp[i] = np.sum( o**2 * mask[:,i])
			if (iter > 1):
				norm=r[i]/sp[i]
				j= np.where(abs(order[:,i]-np.transpose(norm*o))>7.*dev)		
				mask[j,i]=0.0			
				dev_new=dev_new+np.sum(mask[:,i]*(order[:,i]-np.transpose(norm*o))**2)
				
		if (iter >1 ):
			dev=np.sqrt(noise**2+dev_new/np.sum(mask))
			
		if (lamb_sp != 0.0):
			lamda=lamb_sp*np.sum(sp)/ncol
			ab=np.zeros((3,ncol))
			ab[0,1:]=-lamda
			ab[2,:-1]=-lamda
			ab[1,0]=lamda+1.
			ab[1,-1]=lamda+1.
			ab[1,1:-1]=2.*lamda+1.
			sp=solve_banded((1,1), ab, r/sp, overwrite_ab=False, overwrite_b=False)
			
		else:
			sp = r/sp
		
		#Convergence
		if ((abs(sp-sp_old)/sp.max()).max()<0.00001): 
			break
	
	jbad=np.array(0,dtype=np.int64)
	unc=np.repeat(0.,ncol)
	im_out=np.zeros_like((order))
	slitfunc_out=np.zeros_like((order))
	
	#Reconstruction and uncertainties
	for i in range(0, ncol):
		omega=np.repeat(weight,osample)
		yy=y+ycen[i]
		ind1=np.where((yy>=0.0) & (yy<nrow))[0]
		i1=ind1[0]
		i2=ind1[-1]
		omega[0]=yy[i1]
		ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
		o=np.dot(ssf,omega)		
		yyy=nrow-yy[i2]
		o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
		o[nrow-1]=o[nrow-1]+sf[i2]*yyy
		j = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()<5*dev)
		b = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()>=5*dev)
		nj=sp[j].shape[0]
		
		if (nj< nrow):
			jbad=np.append(jbad, nrow*i+b[0])
		if (nj>2):
			ss=np.sum((order[j,i]-sp[i]*o[j])**2)
			xx=np.sum((o[j]-np.mean(o[j]))**2)*(nj-2)
			unc[i]=ss/xx
		else:
			unc[i]=0.0
		im_out[:,i]=np.transpose(sp[i]*o)
		slitfunc_out[:,i]=np.transpose(o)
	
	#Outputs as desired
	if (uncert ==True) and (im_output ==True):
		return(sp,unc,im_out,slitfunc_out)
	elif uncert==True:
		return(sp,unc,slitfunc_out)
	elif im_output==True:
		return(sp,im_out,slitfunc_out)
	else: return(sp,slitfunc_out)
