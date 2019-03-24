! This is a subroutine that finds y from A.y = F for
! banded NON-symmetric matrices. From Thompson text
!
! INPUTS:
!        A  = non-symmetric banded matrix
!        F  = vector from A.Y = F
!        IB = bandwidth of matrix (columns)
!        neq= number of rows in the matrix
!
! OUTPUTS: 
!        Y  = solution vector from A.Y = F
!----------------------------------------------------------
SUBROUTINE nsymgauss(Y,A,F,neq,IB)
  IMPLICIT NONE
  INTEGER :: Idiag,i,j,k,Jend,Kbgn,Kend,Ikc,Jkc,Iback,Jc,Kc
  REAL :: FAC
  INTEGER, INTENT(IN) :: neq,IB
  REAL,DIMENSION(neq,IB) :: AA
  REAL,DIMENSION(neq) :: FF
  REAL,DIMENSION(neq,IB),INTENT(IN) :: A
  REAL,DIMENSION(neq),INTENT(IN) :: F
  REAL,DIMENSION(neq),INTENT(OUT) :: Y

! Initialize matrices to overwrite
  DO i = 1,neq
     DO j = 1,IB
        AA(i,j) = A(i,j)
     END DO
     FF(i) = F(i)
  END DO

! Forward elimination
  Idiag = ((IB - 1)/2) + 1
  DO i = 1,neq-1
     Jend = neq
     IF (Jend > (i+Idiag-1)) Jend = i+Idiag-1
     DO j = i+1,Jend
        Kc   = Idiag - (j-i)
        FAC  = -AA(j,Kc)/AA(i,Idiag)
        Kbgn = i
        Kend = Jend
        DO k = Kbgn,Kend
           Ikc = Idiag + (k-i)
           Jkc = Idiag + (k-j)
           AA(j,Jkc) = AA(j,Jkc) + FAC*AA(i,Ikc)
        END DO
        FF(j) = FF(j) + FAC*FF(i)
     END DO
  END DO

! Backward substitution
  Y(neq) = FF(neq)/AA(neq,Idiag)
  DO Iback = 2,neq
     i    = neq - Iback + 1
     Jend = neq
     IF (Jend > (i+(Idiag-1))) Jend = i+(Idiag-1)
     DO j = i+1,Jend
        Jc    = Idiag + (j-i) 
        FF(i) = FF(i) - AA(i,Jc)*Y(j)
     END DO
     Y(i) = FF(i)/AA(i,Idiag)
  END DO
END SUBROUTINE nsymgauss
