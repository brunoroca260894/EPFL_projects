!----------------------------------------------------------------
!  Lennard-Jones liquid Fortran 90 subroutines
!  f2py compliant  
!  over 100x performance boost than calling pure numpy
!  Wei Chen EPFL 11.2011

!----------------------------------------------------------------
!  calculate lj potential
      subroutine delta(d, d0, rc, iat, eps, sigma, n, du, dv)
      implicit none
      integer :: iat, i, n
      real(8) :: rc, ri, ri0, sigma, eps, u0iat, v0iat, uiat, viat
      real(8), dimension(n) :: diat, d0iat
      real(8), dimension(n, n) :: d, d0
      real(8) :: du, dv
!f2py intent(in) :: d, rc, iat, sigma, n
!f2py intent(out) :: du, dv
        
      diat = d(iat+1,:)
      d0iat = d0(iat+1,:)
      uiat = 0     ! lj potential
      viat = 0     ! 2nd order virial coeff.
      u0iat = 0
      v0iat = 0

      do i = 1, n
        if (i /= iat +1) then
          if (diat(i) <= rc) then
            ri = sigma/diat(i)
            uiat = uiat + ri**12 - ri**6
            viat = viat + ri**12 - 0.5*ri**6
          end if
          if (d0iat(i) <= rc) then
            ri0 = sigma/d0iat(i)
            u0iat = u0iat + ri0**12 - ri0**6
            v0iat = v0iat + ri0**12 - 0.5*ri0**6
          end if
        end if
      du = 4*eps*(uiat-u0iat)
      dv = 48*eps*(viat-v0iat)
      end do
      end

!-------------------------------------------------------------------
!  distance matrix 
      subroutine distmat(atoms, l, n, d)
      implicit none
      integer :: n, i, j
      real(8) :: l
      real(8), dimension(n,3) :: atoms
      real(8), dimension(n,3) :: r
      real(8), dimension(n,n) :: d
!f2py intent(in) :: atoms, l, n
!f2py intent(out) :: d
!f2py intent(hide) :: r

      do i = 1, n
        do j = 1, n
          r(j,:) = abs(atoms(j,:)-atoms(i,:))
        end do
        where (r > 0.5*l) r = r - l
        do j = 1, n
          d(i,j) = r(j,1)**2 + r(j,2)**2 + r(j,3)**2
        end do
      end do
      d = sqrt(d)
      end

!-------------------------------------------------------------------
!  pair distribution function
      subroutine gr(atoms, nbins, l, rho, g, n)
      implicit none
      integer :: nbins, n, i, j
      real(8) :: l, r, dr, rho, pi, p
      real(8), dimension(n,3) :: atoms
      real(8), dimension(n,n) :: d
      integer, dimension(n,n) :: b
      real(8), dimension(nbins,2) :: g
      integer, dimension(nbins) :: hist
      external :: distmat
!f2py intent(in) :: atoms, nbins, n, l, rho
!f2py intent(out) :: g

      pi = 3.1415926536
      dr = 0.5*l/nbins
      hist = 0
      call distmat(atoms, l, n, d) 
      b = ceiling(d/dr)
      do i = 1, n
        do j = 1, n
          if (b(i,j) <= nbins) then
            hist(b(i,j)) = hist(b(i,j)) + 1
          end if
        end do
      end do

      do i = 1, nbins
        r = (i-0.5)*dr
        p = hist(i)/(4*pi*(r**2)*dr)/(rho*n)
        g(i,1) = r
        g(i,2) = p
      end do
      end
