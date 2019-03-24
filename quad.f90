SUBROUTINE QUAD(L1,L2,L3,LWT)
  IMPLICIT NONE
! THIS CREATES THE TABLE OF VALUES FOR TRIANGULAR ELEMENTS
! USED FOR THREE-POINT QUADRATURE
!
!  INTEGRATION REGION:
!
!      0 <= X,  AND  0 <= Y,  AND   X + Y <= 1.
!
!  GRAPH:
!
!      ^
!    1 | *
!      | |\
!    Y | | \
!      | |  \
!    0 | *---*
!      +------->
!        0 X 1
!
!
!
INTEGER :: I
REAL,DIMENSION(3),INTENT(OUT) :: L1,L2,L3,LWT

! INITIALIZE
DO I=1,3
   L1(I)  = 0.0
   L2(I)  = 0.0
   L3(I)  = 0.0
   LWT(I) = 0.0
END DO


! THREE-POINT QUADRATURE
L1(1)  = 0.0
L1(2)  = 0.5
L1(3)  = 0.5
L2(1)  = 0.5
L2(2)  = 0.0
L2(3)  = 0.5
L3(1)  = 0.5
L3(2)  = 0.5
L3(3)  = 0.0
LWT(1) = 1.0/3.0
LWT(2) = 1.0/3.0
LWT(3) = 1.0/3.0
END SUBROUTINE QUAD
