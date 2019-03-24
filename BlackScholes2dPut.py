#!/usr/bin/env python

# Import Modules
#-------------------------------------------------------

# Scipy and Numpy
from scipy import *
import scipy.special as special
# Mesh generator 
import mesh as tri

# Non-symmetric Gaussian Solver
import nsymgauss as NSG

# Quadrature data
import quad as GQD

# Shape function module
import sfntri as SFN

# GUI - wxPython busted at moment
## import femGUI


# Define python functions
#-------------------------------------------------------
def vanilla(r, K, dt, sigma, S):
    """
    This is a simple Vanilla Put Option calcualtion
    based on the analytic solution for a single
    underlying asset. The solution used is from
    The Mathematics of Financial Derivatives, Wilmott, et al.
    Uses ndtr and exp from scipy and ndtr scipy.special modules.

    r    : risk free rate (float)
    K    : strike price (float)
    dt   : time to expiry (float)
    sigma: volatility of S (float)
    S    : range of underlying values (array[float])

    Usage:

    put_value = vanilla(r, K, dt, sigma, S)
    
    """
    
    d1 = zeros(len(S))
    d2 = zeros(len(S))
    n1 = zeros(len(S))
    n2 = zeros(len(S))    
    pt = zeros(len(S))
    b  = sigma*sqrt( dt)
    dsct = exp(-1.0*r*dt)
    for i in range(len(S)):
        d1[i] = (log(S[i]/K) + (r + (0.5* sigma**2))*dt)/b
        d2[i] = (log(S[i]/K) + (r - (0.5* sigma**2))*dt)/b
        n1[i] = special.ndtr(-1.0*d1[i])
        n2[i] = special.ndtr(-1.0*d2[i])    
        pt[i] = K*dsct*n2[i] - S[i]*n1[i]

    return pt


# Create the boundary information arrays only once to
# save calculations during time-dependent BCs 
def findbc(gnodes,s1max,s2max,nnm):
    """
    This function will return an array of values for
    which global nodes lie on the boundaries.

    bnode:
          0 = interior node
          1 = boundary node 

    mind*: the indices of the axes' s1min,s2min nodes
    maxd*: the indices of the axes' s1max,s2max nodes
    
    gnodes is an array of size (num of nodes) x 2
    gnodes[:,0] = global x values
    gnodes[:,1] = global y values

    bnode,mindx,maxdx,mindy,maxdy,s1y0,s2x0
          = findbc(gnodes,s1max,s2max,nnm)
    """

    # The "max" matrices are not actually called in the present
    # program, but may be needed later.
    bnode  = zeros(nnm,dtype=int)
    mxndx  = zeros(nnm,dtype=int)
    mnndx  = zeros(nnm,dtype=int)
    mxndy  = zeros(nnm,dtype=int)
    mnndy  = zeros(nnm,dtype=int)
    for i in range(nnm):
        if allclose(gnodes[i,0],0.0): # axis -> (x=0, y[:])
            bnode[i] = 1              # BC here = vanilla(s2,t)
            mnndx[i] = i
        elif allclose(gnodes[i,0],s1max):
            # axis -> (x=s1max, y[:]) BC here = 0.0
            bnode[i] = 1                   
            mxndx[i] = i
            
    for j in range(nnm):
        if allclose(gnodes[j,1],0.0): # axis -> (x=[:],y=0)
            bnode[j] = 1              # BC here = vanilla(s1,t)
            mnndy[j] = j
        elif allclose(gnodes[j,1],s2max):
            # axis -> (x=[:],y=s2max) BC here = 0.0
            bnode[j] = 1                   
            mxndy[j] = j

    # Create array of only the non-zero entries
    # These are the outer nodes.
    tmp1x = mnndx[mnndx.nonzero()]
    tmp2x = mxndx[mxndx.nonzero()]
    tmp1y = mnndy[mnndy.nonzero()]
    tmp2y = mxndy[mxndy.nonzero()]
    
    # must include the origin 
    origin = 0
    mindx  = sort(append(tmp1x,origin))
    maxdx  = sort(append(tmp2x,origin))
    mindy  = sort(append(tmp1y,origin))
    maxdy  = sort(append(tmp2y,origin))

    # Need these global coords for time dependent BCs on
    # the boundaries. The convention used here is:
    # -> s1 = 0.0 and all S2 is the y-axis 
    # -> s2 = 0.0 and all S1 is the x-axis    
    s1y0  = zeros(len(mindy),dtype=float)
    s2x0  = zeros(len(mindx),dtype=float)

    # These are the actual global coordinates of the
    # outer nodes. These are required for the BC
    # calculation
    for i in range(len(mindy)):
        s1y0[i] = gnodes[mindy[i],0]

    for i in range(len(mindx)):
        s2x0[i] = gnodes[mindx[i],1]    
    
    return bnode, mindx, maxdx, mindy, maxdy, s1y0, s2x0



# Create inital value (actually the "final" sol'n here)
def initialVal(K,gnodes,nnm,etype):
    u0   = zeros(nnm,dtype=float)
    for i in range(nnm):

        # Toggle the two definitions below for
        # a different exit strategies
        # See ACHDOU & PIRONNEAU eqn's [2.64] & [2.65]
        # You get very different graphs
        if allclose(etype,1.0):
            # [2.64]
            s1s2  = gnodes[i,0] + gnodes[i,1]
        else:
            # [2.65]
            s1s2  = max(gnodes[i,0], gnodes[i,1])

        test  = K - s1s2
        u0[i] = max(test,0.0)

    return u0
            

# Create the function to update the time-dependent BCs
def newBound(nnm,mindx,mindy,r,K,vol1,vol2,dt,s1y0,s2x0):
    newBC = zeros(nnm,dtype=float)

    # Call the vannila PUT function and use outer
    # nodal values
    s1bc = vanilla(r,K,dt,vol1,s1y0)
    s2bc = vanilla(r,K,dt,vol2,s2x0)

    # Set the values equal to the output from vanilla
    # NOTE: I assume that the min of S1 and S2 are at the
    # origin and that they are equal because they share the
    # same strike. The rest are zeros
    for i in range(len(s1bc)):
        bnod = mindy[i]
        newBC[bnod] = s1bc[i]
        
    for i in range(len(s2bc)):
        bnod = mindx[i]
        newBC[bnod] = s2bc[i]

    return newBC 


# Get user input
#-------------------------------------------------------
## app = femGUI.MyApp(False)
## app.MainLoop()
## inputs = femGUI.values

## # Multiply everything by 1.0 or 1 to ensure we have SciPy dtype
## # floats or integers as the GUI passes UNICODE STRINGS!!!
## etype  = float(inputs[0])*1.0
## s1high = float(inputs[1])*1.0
## s2high = float(inputs[2])*1.0
## vol1   = float(inputs[3])*1.0
## vol2   = float(inputs[4])*1.0
## rate   = float(inputs[5])*1.0
## pcorr  = float(inputs[6])*1.0
## K      = float(inputs[7])*1.0
## lastT  = float(inputs[8])*1.0
## dt     = float(inputs[9])*1.0
## nx     = int(inputs[10])*1
## ny     = int(inputs[11])*1
## 

# Below for comparison purposes 
# Comment out GUI inputs and run with the below values
#
# These values are the same used by ACHDOU & PIRONNEAU
# for the creation of Figures [4.11] and [4.12]
# These are the equivalent values for their THETA matrix
# using this formulation
# 
# CHANGE ETYPE FOR THE DIFFERENT FINAL CONDITIONS
# Use either 1.0 or 0.0
etype  = 0.0
s1high = 150.0
s2high = 150.0
vol1   = 0.1414
vol2   = 0.1414
rate   = 0.1
pcorr  = -0.6
K      = 100.0
lastT  = 0.70
dt     = 0.01
nx     = 50
ny     = 50
# Specify zero as the minimum value for the grid. 
s1low  = 0.0
s2low  = 0.0

# Initialize vectors/matrices
# Integer values for loops/sizes
nex1 = nx + 1
ney1 = ny + 1
nem  = 2*nx*ny
nnm  = nex1*ney1
npe  = 3
ndf  = 1
neq  = nnm*ndf
nn   = npe*ndf

# Number of quadrature points
nipf = 3

# Floats and arrays
x0 = s1low
y0 = s2low
dx = ones(nex1,float)*float((s1high/nx))
dy = ones(ney1,float)*float((s2high/ny))
dx[-1] = 0.0
dy[-1] = 0.0

# Create the differential eqn's coefficients
f0   = 0.0
c0   = 1.0
a110 = 0.5*(vol1**2.0)
a220 = 0.5*(vol2**2.0)
a120 = pcorr*vol1*vol2
b10  = rate
b20  = rate
G    = -1.0*rate

# Call Fortran Mesh routine
# NOTA BENE: The connectivity matirx NODF has indices
#            according to the FORTRAN CONVENTION!
nodf,glxy = tri.mesh(nx,ny,nex1,ney1,nem,nnm,dx,dy,x0,y0)

# Switch NODF indices for the Python convention
fort2py   = ones(shape(nodf),dtype=int)
nodp      = nodf - fort2py

# Find IdaigF and Idiag where they are the Fortran and Python 
# index of the diagonal for the non-symmetric stiffness matrix 
# respectively -> RECALL: Python starts indexing at 0!
IdiagF = 0 
for i in range(nem):
    for j in range(npe):
        for k in range(npe):
            nw = (int(abs(nodf[i,j] - nodf[i,k])+1))*ndf
            if IdiagF < nw:
                IdiagF = nw

# Band width of sparse matrix
band  = (IdiagF*2) - 1
Idiag = IdiagF - 1


#-------------------------------------------------------#
#                                                       #
#                   Begin FEM Routine                   #
#                                                       #
#-------------------------------------------------------#


# [1] Set time values
# Time dependent variables & Crank-Nicolson parameters
alfa   = 0.5
ntime  = int(lastT/dt) + 1
a1     = alfa*dt
a2     = (1.0 - alfa)*dt

# Create storage matrices for values at each time step
optionValue = zeros((nnm,ntime),dtype=float)
optionParam = zeros((nnm,ntime),dtype=float)

# [2] Initialize BCs

# Create "final" condition and store for option price calculation
# once all the values in time have been calculated
u0   = initialVal(K,glxy,nnm,etype)
glu  = u0

# Generate boundary information matrices from global matrix
bnode,mindx,maxdx,mindy,maxdy,s1y0,s2x0 = \
                               findbc(glxy,s1high,s2high,nnm)

# An array of Python indices
nwld = arange(nnm,dtype=int)

# [3] Enter time loop
time   = 0.0
ncount = 0
while ncount < ntime :
    
    # Find new BCs for future time step
    time += dt
    newBC = \
      newBound(nnm,mindx,mindy,rate,K,vol1,vol2,time,s1y0,s2x0)

    # Global matrices
    glk  = zeros((neq,band),dtype=float)
    glf  = zeros(neq,dtype=float)

    # Begin loop over each element
    for n in range(nem):
        # Element matrices
        elxy = zeros((npe,2),dtype=float)
        elu  = zeros(npe,dtype=float)    
        elf  = zeros(npe,dtype=float)
        elm  = zeros((npe,npe),dtype=float)
        elk  = zeros((npe,npe),dtype=float)
        
        for i in range(npe):
            # Assign global values for each node in the element
            ni = nodp[n,i]
            elxy[i,0] = glxy[ni,0]
            elxy[i,1] = glxy[ni,1]            
            elu[i]    = glu[ni]
                
        # [4] Now compute elemental matrices
        # Load quadrature data from Fortran Module       
        l1,l2,l3,lwt = GQD.quad() 

        # [5] Begin quadtrature loop
        for nl in range(npe):
            ac1 = l1[nl]
            ac2 = l2[nl]
            ac3 = l3[nl]
            
            # Call Fortran Shape Function Module
            det,sf,gdsf = SFN.sfntri(ac1,ac2,ac3,elxy)            
            cnst = 0.5*det*lwt[nl]

            # Global x an y in terms of the unit triangle
            x = 0.0
            y = 0.0
            for it in range(npe):
                x += elxy[it,0]*sf[it]
                y += elxy[it,1]*sf[it]

            # Set coefficients with mapped x and y coordinates
            a11 = a110*x*x
            a22 = a220*y*y
            a12 = a120*x*y 
            b1  = b10*x 
            b2 = b20*y 
            source = f0
            ct     = c0            

            # Create Elemental K, M, and F matrices/vector
            # by integrating over the element
            for ip in range(npe):
                for jp in range(npe):
                    s00 = sf[ip]*sf[jp]*cnst
                    s11 = gdsf[0,ip]*gdsf[0,jp]*cnst
                    s22 = gdsf[1,ip]*gdsf[1,jp]*cnst
                    s12 = gdsf[0,ip]*gdsf[1,jp]*cnst
                    s01 = sf[ip]*gdsf[0,jp]*cnst
                    s02 = sf[ip]*gdsf[1,jp]*cnst
                    # Now assemble ELEMENT MATRIX [K]
                    # using the form from THOMPSON
                    # [K]  = [S1] - [S2] - [S3] - [Sh] where
                    # [Sh] = 0.0 for this problem
                    elk[ip,jp] += (a11*s11 + a12*s12 + a22*s22)\
                                  - (b1*s01 + b2*s02) \
                                  - G*s00                     
                    elm[ip,jp] += ct*s00                        
                elf[ip] += cnst*sf[ip]*source                   
    
        # [6] Apply CRANK-NICOLSON to find K^ and F^
        # See J.N. REDDY, eqn (6.42b)
        for ik in range(nn):
            summ = 0.0
            for jk in range(nn):
                summ += (elm[ik,jk] - a2*elk[ik,jk])*elu[jk]  
                elk[ik,jk] = elm[ik,jk] + a1*elk[ik,jk]
            elf[ik] = (a1+a2)*elf[ik] + summ        

        # [7] Assemble into global matrices using the
        # routine from THOMPSON for banded & non-symmetric
        for j in range(npe):
            jnp = nodp[n,j]
            jeq = nwld[jnp]
            glf[jeq] += elf[j]
            for k in range(npe):
                knp = nodp[n,k]
                keq = nwld[knp]
                kb  = (keq-jeq) + Idiag
                glk[jeq,kb] += elk[j,k]    
    
    # [8] Apply BCs by BLASTING technique (also a THOMPSON thing)
    BLAST = 1.0e6
    for i in range(nnm):
        if allclose(bnode[i],1):
            nb = nwld[i]
            glu[nb] = newBC[i]               
            glk[nb,Idiag] *= BLAST
            glf[nb] = glu[i]*glk[nb,Idiag]
      
    # [9] Solve GLOBAL MATRICES using Fortran nsymgauss module
    glu = NSG.nsymgauss(glk,glf,neq,band)        

    # [10] Store data for visualization at the end
    oValu = glu
    oPara = glu - u0
    for i in range(nnm):
        optionValue[i,ncount] = oValu[i]
        optionParam[i,ncount] = oPara[i]

    # [11] Update the time loop and BCs for next time step
    ncount += 1
    
# END OF TIME LOOP HERE --------------------------------


# Visualize with MayaVi
# UPDATE: 2014-05-17
# PyVTK and MayaVI1 no longer the supported versions
# TVTK and MayaVI2 are the new world and the code
# below will bomb. 
#
# For now, I recommend the Matplotlib versions
#-------------------------------------------------------

# Set the z1 variable (second column is time)
####z1 = u0
## z1 = optionValue[:,-1]
####z1 = optionParam[:,-1]
## 
## import pyvtk
## # Scale the data in the Z-direction
## dzz = dx[0]*2
## dxx = dx[0]
## dyy = dy[0]
## 
## # Convert z1 to vtk structured point data
## # Note: No need to rearrange z1 as it is already in the
## #       proper sequence from the meshing routine
## point_data = pyvtk.PointData(pyvtk.Scalars(z1))
## 
## # Generate the grid sizing
## grid = pyvtk.StructuredPoints((nex1,nex1,1),(0,0,0),(dxx,dyy,dzz))
## 
## # Save to temporary file
## data = pyvtk.VtkData(grid, point_data)
## data.tofile('/tmp/test.vtk')
## 
## # Now use MayaVi to visualize
## import mayavi
## v = mayavi.mayavi() # create a MayaVi window.
## d = v.open_vtk('/tmp/test.vtk', config=0) # open the data file.
## 
## # Load the filters.
## f = v.load_filter('WarpScalar', config=0) 
## n = v.load_filter('PolyDataNormals', 0)
## n.fil.SetFeatureAngle (45)
## 
## # Load the necessary modules.
## m = v.load_module('SurfaceMap', 0)
## a = v.load_module('Axes', 0)
## t = v.load_module('Text',0)
## o = v.load_module('Outline', 0)
## 
## # Re-render the scene.
## v.Render() 
## v.master.wait_window()


# Or visualize with Matplotlib
#-------------------------------------------------------

# An alternative option (for speed) is matplotlib
# Output data to figures using matplotlib below
import pylab as p
#import matplotlib.axes3d as p3
from mpl_toolkits.mplot3d import axes3d
x  = reshape(glxy[:,0],(nex1,ney1))
y  = reshape(glxy[:,1],(nex1,ney1))
init_val = u0
finalval = optionValue[:,-1]
time_val = optionParam[:,-1]
z1 = reshape(init_val,(nex1,ney1))
z2 = reshape(finalval,(nex1,ney1))
z3 = reshape(time_val,(nex1,ney1))

# Make three figures
fig1= p.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(x,y,z1)
ax1.set_xlabel('S1')
ax1.set_ylabel('S2')
ax1.set_zlabel('Final Condition at Expiry')

fig2= p.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_wireframe(x,y,z2)
ax2.set_xlabel('S1')
ax2.set_ylabel('S2')
ax2.set_zlabel('Option Value')

fig3= p.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_wireframe(x,y,z3)
ax3.set_xlabel('S1')
ax3.set_ylabel('S2')
ax3.set_zlabel('Time Value')

# Show the plots: NOTE that you can rotate them
# with a mouse
p.show()
