
library(deSolve)

# TODO: test with different models

# model
X_0 <- function(th) matrix(th[1])
P0 <- function(th) matrix(th[2])
Phi <- function(th) matrix(th[3])
Psi <- function(th) matrix(th[4])
G <- function(th) matrix(th[5])
H <- function(th) matrix(th[6])
Q <- function(th) matrix(th[7])
R <- function(th) matrix(th[8])

# define th, u, y

# loglik
L <- function(Phi, Psi, G, Q, H, R, X_0, P0, th, u, y, t)
{
	inv <- function(A) solve(A)
	
	Phi <- Phi(th)
	G <- G(th)
	Psi <- Psi(th)
	Xe <- X_0(th)
	Pe <- P0(th)
	Q <- Q(th)
	H <- H(th)
	R <- R(th)

	# 1
	j <- 2 # j starts from 2, not 1, because in R vectors indices are unity-based
	chi <- 0
	N <- length(t)
	m <- ncol(H)
	n <- nrow(Phi)
	I <- diag(n)

	dXp <- function(t, Xp, parms=NULL)
		list(c(Phi %*% Xp + Psi %*% u(t)))

	dPp <- function(tt, Pp, parms=NULL) {
		Pp <- matrix(Pp, n, n)
		list(c(Phi %*% Pp %*% t(Phi) + G %*% Q %*% t(G)))
	}

	for (j in 2:N)
	{
		# 2
	  tt <- c(t[j-1], t[j])

	  Xp <- ode(c(Xe), tt, dXp)
		Xp <- Xp[nrow(Xp),] # get last row
		Xp <- Xp[-1] # throw away time value

	  Pp <- ode(c(Pe), tt, dPp)
		Pp <- Pp[nrow(Pp),] # get last row
		Pp <- Pp[-1] # throw away time value
		Pp <- matrix(Pp, n, n) # form matrix

	  # 3
	  e <- y[j-1] - H %*% Xp
	  B <- H %*% Pp %*% t(H) + R
	  invB <- inv(B)
	  K <- Pp %*% t(H) %*% invB

	  # 4
	  S <- 1/2 * (t(e) %*% invB %*% e + log(det(B)))

		# 5
		chi <- chi + S

	  # 6
		Xe <- Xp + K %*% e
		Pe <- (I - K %*% H) %*% Pp
	}
	chi <- chi + N*m / 2 * log(2*pi)
	return(chi)
}
