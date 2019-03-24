! This is the 2D mesh generator for linear, triangular elements
! 
! This code is modified from J.N. REDDY, AN INTRODUCTION TO THE
! FINITE ELEMENT METHOD, 2nd EDITION
! 
! Coded by Tyler Hayes, April 3, 2007
!
!
! INPUTS:
!
!      NX = NUMBER OF DIVISIONS IN THE X-DIR
!      NY = NUMBER OF DIVISIONS IN THE Y-DIR
!    NEX1 = INTEGER = NX+1
!    NEY1 = INTEGER = NY+1
!     NEM = INTEGER = 2*NX*NY
!     NNM = INTEGER = ((IEL*NX)+1) * ((IEL*NY)+1) IEL = 1 for tri
!      DX = VECTOR OF SPACINGS IN THE X-DIR <- CAN BE VARIABLE
!           THE VECTOR SHOULD BE NX+1 IN SIZE BUT THE DX(NX+1) IS
!           SET TO ZERO AND IS A DUMMY VARIABLE
!      DY = VECTOR OF SPACINGS IN THE Y-DIR <- CAN BE VARIABLE
!           THE VECTOR SHOULD BE NY+1 IN SIZE BUT THE DY(NY+1) IS
!           SET TO ZERO AND IS A DUMMY VARIABLE
!      X0 = ORIGIN OF THE X AXIS
!      Y0 = ORIGIN OF THE Y AXIS
!
! OUTPUT:
!      NOD = INTEGER MATRIX OF ELEMENT NOD INDICES
!     GLXY = GLOBAL COORDINATES OF NOD
!
! MISC.:
!      NPE = NODES PER ELEMENT
!----------------------------------------------------------------

SUBROUTINE MESH(NOD,GLXY,NX,NY,NEX1,NEY1,NEM,NNM,DX,DY,X0,Y0)
  IMPLICIT NONE
  INTEGER :: NXX,NYY,NX2,NY2,NXX1,NYY1
  INTEGER :: K,IY,L,M,N,I,NI,NJ,IEL
  INTEGER,PARAMETER :: NPE=3
  REAL :: XC,YC
! INPUTS
  INTEGER,INTENT(IN) :: NX,NY,NEM,NNM,NEX1,NEY1
  REAL,INTENT(IN) :: X0,Y0
  REAL,DIMENSION(NEX1),INTENT(IN) :: DX
  REAL,DIMENSION(NEY1),INTENT(IN) :: DY
! PARAMETERS
! OUTPUTS
  INTEGER,DIMENSION(NEM,NPE),INTENT(OUT) :: NOD
  REAL,DIMENSION(NNM,2),INTENT(OUT) :: GLXY

! CREATE VARIABLES
  IEL  = 1 ! FOR <= 4 NODES PER ELEMENT
  NXX  = IEL*NX
  NYY  = IEL*NY
  NXX1 = NXX + 1
  NYY1 = NYY + 1
  NX2  = 2*NX
  NY2  = 2*NY


! CREATE TRIANGUALR ELEMENTS
!----------------------------------------------------------------

! INITIALIZE FIRST TWO ELEMENTS
  NOD(1,1) = 1
  NOD(1,2) = IEL+1
  NOD(1,3) = IEL*NXX1 + IEL + 1
  NOD(2,1) = 1
  NOD(2,2) = NOD(1,3)
  NOD(2,3) = IEL*NXX1 + 1

! LOOP THROUGH MESH
  K=3
  DO IY=1,NY
     L=IY*NX2
     M=(IY-1)*NX2
     IF (NX > 1) THEN
        DO N=K,L,2
           DO I=1,NPE
              NOD(N,I)   = NOD(N-2,I) + IEL
              NOD(N+1,I) = NOD(N-1,I) + IEL
           END DO
        END DO
     END IF
     IF (IY < NY) THEN
        DO I=1,NPE
           NOD(L+1,I) = NOD(M+1,I) + IEL*NXX1
           NOD(L+2,I) = NOD(M+2,I) + IEL*NXX1
        END DO
     END IF
     K=L+3
  END DO


! NOW GENERATE GLOBAL COORDINATES OF THE NODES
  XC = X0
  YC = Y0

  DO NI=1,NEY1
     XC = X0
     I  = NXX1*IEL*(NI-1)
     DO NJ = 1,NEX1
        I=I+1
        GLXY(I,1) = XC
        GLXY(I,2) = YC
        IF (NJ < NEX1) THEN
           IF (IEL == 2) THEN
              I=I+1
              XC = XC + 0.5*DX(NJ)
              GLXY(I,1) = XC
              GLXY(I,2) = YC
           END IF
        END IF
        XC = XC + DX(NJ)/IEL
     END DO
     XC = X0
     IF (IEL == 2) THEN
        YC = YC + 0.5*DY(NI)
        DO NJ = 1,NEX1
           I=I+1
           GLXY(I,1) = XC
           GLXY(I,2) = YC
           IF (NJ < NEX1) THEN
              I=I+1
              XC = XC + 0.5*DX(NJ)
              GLXY(I,1) = XC
              GLXY(I,2) = YC
           END IF
           XC = XC + 0.5*DX(NJ)
        END DO
     END IF
     YC = YC + DY(NI)/IEL
  END DO
END SUBROUTINE MESH
  
