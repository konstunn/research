
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

inv <- function(A) solve(A)
Sp <- function(A) sum(diag(A))

# loglik
L <- function(Phi, Psi, G, Q, H, R, X_0, P0, th, u, y, t)
{
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
		Xp <- tail(Xp, n=1)
		Xp <- Xp[-1] # throw away time value

	  Pp <- ode(c(Pe), tt, dPp)
		Pp <- tail(Pp, n=1)
		Pp <- Pp[-1] # throw away time value
		Pp <- matrix(Pp, n, n) # form matrix

	  # 3
	  e <- y[j] - H %*% Xp
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

library(numDeriv)
# returns list of partial derivatives of matrices by theta_i
# A - matrix-function
matderiv <- function(A, th)
{
	dA <- jacobian(A, th)

	A <- as.matrix(A(th))

	f <- function(a) list(matrix(a, nrow=nrow(A), ncol=ncol(A)))

	dA <- apply(dA, 2, f)

	dA <- Reduce(c, dA)

	return(dA)
}

library(Matrix)

gradL <- function(Phi, Psi, G, Q, H, R, X_0, P0, th, u, y, t)
{
	s <- length(th)
	N <- length(t)
	n <- nrow(Phi)

	dPhi <- matderiv(Phi, th)
	dPsi <- matderiv(Psi, th)
	dG <- matderiv(G, th)
	dQ <- matderiv(G, th)
	dH <- matderiv(H, th)
	dR <- matderiv(R, th)

	Phi <- Phi(th)
	Psi <- Psi(th)
	G <- G(th)
	Q <- Q(th)
	H <- H(th)
	R <- R(th)

	dX_0 <- matderiv(X_0, th)
	X_0 <- X_0(th)

	dP_0 <- matderiv(P_0, th)
	P_0 <- P_0(th)

	# Phi_A
	Phi_dPhi <- rbind(Phi, Reduce(rbind, dPhi))
	block_diag <- replicate(s, Phi, simplify=FALSE)
	block_diag <- bdiag(block_diag)
	O <- matrix(0, n, n)
	O <- replicate(s, O, simplify=FALSE)
	O <- Reduce(cbind, O)
	Phi_A <- cbind(Phi_dPhi, rbind(O, block_diag))

	# Psi_dPsi
	Psi_dPsi <- rbind(Psi, Reduce(rbind, dPsi))

	d_Xp_dXp <- function(tt, X_dX, parms=NULL)
	{
		X_dX <- matrix(X_dX, nrow=n*(s+1))
		d_X_dX <- Phi_A %*% X_dX + Psi_dPsi %*% u(tt)
		list(c(d_X_dX))
	}

	# Phi_dPhi_t
	dPhi_t <- lapply(dPhi, t)
	dPhi_t <- Reduce(rbind, dPhi_t)
	Phi_dPhi_t <- rbind(t(Phi), dPhi_t)

	# G_dG
	G_dG <- rbind(G, Reduce(rbind, dG))

	# O_dG_t
	dG_t <- lapply(dG, t)
	dG_t <- Reduce(rbind, dG_t)
	O <- matrix(0, nrow(G), ncol(G))
	O_dG_t <- rbind(O, dG_t)

	# O_dQ
	O <- matrix(0, nrow(Q), ncol(Q))
	O_dQ <- rbind(O, Reduce(rbind, dQ))

	d_Pp_dPp <- function(tt, P_dP, parms=NULL)
	{
		P_dP <- matrix(P_dP, nrow=n*(s+1))

		P <- P_dP[1:n,]
		block_diag <- replicate(s, P, simplify=FALSE)
		block_diag <- bdiag(P_A)

		O <- matrix(0, n, n)
		O <- replicate(s, O)
		O <- Reduce(cbind, O)

		P_A <- rbind(O, block_diag)
		P_A <- cbind(P_dP, P_A)

		d_P_dP <- Phi_A %*% P_dP + P_A %*% Phi_dPhi_t + G_dG %*% Q %*% t(G) +
			G %*% O_dQ %*% t(G) + G %*% Q %*% O_dG_t

		list(c(d_P_dP))
	}

	# 1
	dL <- as.list(rep(0, s))

	# 2
	for (j in 2:N) {
		if (j == 2) {
			Xe_dXe <- Reduce(rbind, dX_0)
			Xe_dXe <- rbind(X_0, Xe_dXe)

			Pe_dPe <- Reduce(rbind, dP_0)
			Pe_dPe <- rbind(P_0, Pe_dPe)
		} else {
			Xe <- Xp + K %*% E

			dK_E <- Map('%*%', dK, E)
			K_dE <- Map('%*%', K, dE)

			dXe <- Map('+', dK_E, K_dE)
			dXe <- Map('+', dXe, dXp)

			Xe_dXe <- Reduce(rbind, dXe)
			Xe_dXe <- rbind(Xe, Xe_dXe)

			I <- diag(n)
			Pe <- (I - K %% H) %*% Pp

			dK_H <- Map('%*%', dK, -H)
			K_dH <- Map('%*%', -K, dH)
			I_K_H_dPp <- Map('%*%', I - K %*% H, dPp)
			dK_H_K_dH <- Map('+', dK_H, K_dH)
			dK_H_K_dH_Pp <- Map('%*%', dK_H_K_dH, Pp)
			dPe <- Map('+', dK_H_K_dH_Pp, I_K_H_dPp)

			Pe_dPe <- Reduce(rbind, dPe)
			Pe_dPe <- rbind(Pe, Pe_dPe)
		}

		tt <- c(t[j-1], t[j])

		Xp_dXp <- ode(c(Xe_dXe), tt, d_Xp_dXp)

		Xp_dXp <- tail(Xp_dXp, n=1)
		Xp_dXp <- Xp_dXp[-1] # throw away first element (time value)

		Xp <- head(Xp_dXp, n)
		dXp <- tail(Xp_dXp, -n)

		Pp_dPp <- ode(c(Pe_dPe), tt, d_Pp_dPp)

		Pp_dPp <- tail(Pp_dPp, n=1)
		Pp_dPp <- Pp_dPp[-1] # throw away first element (time value)
		Pp_dPp <- matrix(Pp_dPp, n*(s+1))

		Pp <- head(Pp_dPp, n)
		dPp <- tail(Pp_dPp, -n)

		# 3
		B <- H %*% Pp %*% t(H) + R
		invB <- inv(B)

		# dB
		dH_Pp_Ht <- Map('%*%', dH, Pp %*% t(H))
		H_dPp_Ht <- lapply(dPp, function(dPp_i) H %*% Pp_i %*% t(H))
		H_Pp_dHt <- Map('%*%', H %*% Pp, t(dH))

		dB <- Map('+', dH_Pp_Ht, H_dPp_Ht)
		dB <- Map('+', dB, H_Pp_dHt)
		dB <- Map('+', dB, dR)

		# 4
		K <- Pp %*% t(H) %*% invB

		#dK
		dPp_Ht_invB <- Map('%*%', dPp, t(H) %*% invB)
		Pp_dHt_invB <- lapply(dH, function(dH_i) Pp %*% t(dH_i) %*% invB)
		Pp_Ht_invB_dB_invB <- Map('%*%', Pp %*% t(H) %*% invB, dB)
		Pp_Ht_invB_dB_invB <- Map('%*%', Pp_Ht_invB_dB_invB, invB)

		dK <- Map('+', dPp_Ht_invB, Pp_dHt_invB)
		dK <- Map('-', dK, Pp_Ht_invB_dB_invB)

		# 5
		E <- y[j] - H %*% Xp

		# dE
		dH_Xp <- Map('%*%', dH, -Xp)
		Xp_dH <- Map('%*%', Xp, -dH)
		dE <- Map('+', dH_Xp, Xp_dH)

		# 6
		# Sk
		invB_dB <- Map('%*%', invB, dB)
		Et_invB_dE <- Map('%*%', t(E) %*% invB, dE)
		Et_invB_dB <- Map('%*%', t(E), invB_dB)
		Et_invB_dB_invB_E <- Map('%*%', Et_invB_dB, -.5 * invB %*% E)
		Sp_invB_dB <- Map(Sp, .5 * invB_dB)

		Sk <- Map('+', Et_invB_dE, Et_invB_dB_invB_E)
		Sk <- Map('+', Sk, Sp_invB_dB)

		# 7
		dL <- Map('+', dL, Sk)

		# 8
	}
	return(dL)
}
