
library(deSolve)
library(numDeriv)
library(Matrix)
library(MASS)

# TODO: test with different models

# model
X_0 <- function(th) matrix(c(0,0))
P0 <- function(th) matrix(c(0.01,0,0,0.01), 2)
Phi <- function(th) matrix(c(th[1],0,0,th[2]), 2)
Psi <- function(th) matrix(c(1,0,0,1), 2)
G <- function(th) matrix(c(0.04,0,0,0.04), 2)
H <- function(th) matrix(c(1,0,0,1), 2)
Q <- function(th) matrix(c(0.01,0,0,0.01), 2)
R <- function(th) matrix(c(0.02,0,0,0.02), 2)

isStable <- function(model, th)
{
	with(model,
	{
		all(Re(eigen(Phi(th))$values) < 0)
	})
}

model <- list(Phi=Phi, Psi=Psi, G=G, H=H, Q=Q, R=R, X_0=X_0, P0=P0)

isConformable <- function(model, th)
{
	model <- lapply(model, function(A) A(th))
	out <- tryCatch({
		with(model,
		{
			# th <- rep(0, nrow(Phi))
			Phi(th) %*% X_0(th) + G(th) %*% matrix(rep(1, nrow(Q(th))))
			Psi(th) %*% X_0(th) + matrix(rep(2, nrow(R(th))))
		})}, 
		error=function(e) e)
	!any(class(out) == 'error')
}

observability <- function(A, B)
{
	n <- nrow(A)
	OM <- B
	for (i in 1:(n-1)) {
		M <- replicate(i, A, simplify=FALSE)
		M <- Reduce('%*%', M)
		M <- M %*% B
		OM <- rbind(OM, M)
	}
	OM
}

isObservable <- function(A, B)
{
	n <- nrow(A)
	OM <- observability(A, B)
	qr(OM)$rank == n
}

# define th, u, y
th <- -(10*runif(2))
t <- seq(0, 10, length.out=100)
u <- function(t) matrix(c(10*sin(2*pi*100*t), 10*cos(2*pi*100*t)))
u <- function(t) matrix(c(10,10))
y <- matrix(c(rnorm(length(t)), rnorm(length(t))))

inv <- function(A) solve(A)
Sp <- function(A) sum(diag(A))

dX <- function(t, Xp, parms)
{
	with(parms,
	{
		n <- nrow(Phi)
		if (length(parms$G) == 0)
			G <- matrix(0, n, n)
		if (length(parms$w) == 0)
			w <- matrix(0, n)
		Xp <- matrix(Xp, n)
		list(c(Phi %*% Xp + Psi %*% u(t) + G %*% w))
	})
}

sim <- function(model, th, t, u)
{
	model <- lapply(model, function(A) A(th))
	yc <- with(model,
	{
		m <- nrow(H)
		n <- nrow(Phi)
		p <- ncol(G)
		k <- length(t)

		Sigma <- bdiag(list(P0, Q, R))
		mv <- mvrnorm(k, c(X_0, rep(0, p), rep(0, m)), Sigma)
		mv <- t(mv)

		X <- mv[1:n, 1]
		mv <- mv[-(1:n), ]

		w <- mv[1:p, ]
		mv <- mv[-(1:p), ]

		v <- mv

		yc <- NULL

		for (i in 2:k) {
			tt <- c(t[i-1], t[i])
			# TODO: return true and noisy state
			X <- ode(c(X), tt, dX, list(Phi=Phi, Psi=Psi, G=G, u=u, w=w[, i]))
			X <- tail(X, n=1)
			X <- X[-1]
			# TODO: return true and noisy observations
			yt <- H %*% X + v[, i]
			yc <- cbind(yc, yt)
		}
		yc
	})
	yc
}

# loglik
L <- function(model, th, u, y, t)
{
	model <- lapply(model, function(A) A(th))

	with(model,
	{
		Xe <- X_0
		Pe <- P0

		# 1
		chi <- 0
		N <- length(t)
		m <- ncol(H)
		n <- nrow(Phi)
		I <- diag(n)

		dPp <- function(tt, Pp, parms=NULL) {
			Pp <- matrix(Pp, n, n)
			list(c(Phi %*% Pp %*% t(Phi) + G %*% Q %*% t(G)))
		}

		# j starts from 2, not 1, because in R vectors indices are unity-based
		for (j in 2:N)
		{
			# 2
			tt <- c(t[j-1], t[j])

			Xp <- ode(c(Xe), tt, dX, list(Phi=Phi, Psi=Psi, u=u))
			Xp <- tail(Xp, n=1)
			Xp <- Xp[-1] # throw away time value

			Pp <- ode(c(Pe), tt, dPp)
			Pp <- tail(Pp, n=1)
			Pp <- Pp[-1] # throw away time value
			Pp <- matrix(Pp, n, n) # form matrix

			# 3
			# browser()
			e <- y[, j-1] - H %*% Xp
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
	})
}

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


# FIXME:
dL <- function(model, th, u, y, t)
{
	with(model,
	{
		s <- length(th)
		N <- length(t)
		n <- nrow(Phi(th))

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

		dP0 <- matderiv(P0, th)
		P0 <- P0(th)

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
			list(as.numeric(d_X_dX))
		}

		# Phi_dPhi_t
		dPhi_t <- lapply(dPhi, t)
		dPhi_t <- Reduce(rbind, dPhi_t)
		Phi_dPhi_t <- rbind(t(Phi), dPhi_t)

		# GdG
		GdG <- c(list(G), dG)

		# OdGt
		dG_t <- lapply(dG, t)
		#dG_t <- Reduce(rbind, dG_t)
		O <- matrix(0, nrow(G), ncol(G))
		OdGt <- c(list(O), dG_t)

		# OdQ
		O <- matrix(0, nrow(Q), ncol(Q))
		OdQ <- c(list(O), dQ)

		d_Pp_dPp <- function(tt, P_dP, parms=NULL)
		{
			P_dP <- matrix(P_dP, nrow=n*(s+1))

			P <- P_dP[1:n,]
			block_diag <- replicate(s, P, simplify=FALSE)
			block_diag <- bdiag(block_diag)

			O <- matrix(0, n, n)
			O <- replicate(s, O, simplify=FALSE)
			O <- Reduce(cbind, O)

			P_A <- rbind(O, block_diag)
			P_A <- cbind(P_dP, P_A)

			#
			GdG_Q_Gt <- Map('%*%', GdG, list(Q %*% t(G)))
			GdG_Q_Gt <- Reduce(rbind, GdG_Q_Gt)

			G_OdQ <- Map('%*%', list(G), OdQ)
			G_OdQ_Gt <- Map('%*%', G_OdQ, list(t(G)))
			G_OdQ_Gt <- Reduce(rbind, G_OdQ_Gt)

			G_Q_OdGt <- Map('%*%', list(G %*% Q), OdGt)
			G_Q_OdGt <- Reduce(rbind, G_Q_OdGt)

			d_P_dP <- Phi_A %*% P_dP + P_A %*% Phi_dPhi_t + GdG_Q_Gt + G_OdQ_Gt +
				G_Q_OdGt

			list(as.numeric(d_P_dP))
		}

		# 1
		dL <- as.list(rep(0, s))

		# 2
		for (j in 2:N) {
			if (j == 2) {
				Xe_dXe <- Reduce(rbind, dX_0)
				Xe_dXe <- rbind(X_0, Xe_dXe)

				Pe_dPe <- Reduce(rbind, dP0)
				Pe_dPe <- rbind(P0, Pe_dPe)
			} else {
				Xe <- Xp + K %*% E

				dK_E <- Map('%*%', dK, list(E))
				K_dE <- Map('%*%', list(K), dE)

				dXe <- Map('+', dK_E, K_dE)
				dXe <- Map('+', dXe, dXp)

				Xe_dXe <- Reduce(rbind, dXe)
				Xe_dXe <- rbind(Xe, Xe_dXe)

				I <- diag(n)
				Pe <- (I - K %*% H) %*% Pp

				dK_H <- Map('%*%', dK, list(-H))
				K_dH <- Map('%*%', list(-K), dH)
				I_K_H_dPp <- Map('%*%', list(I - K %*% H), dPp)
				dK_H_K_dH <- Map('+', dK_H, K_dH)
				dK_H_K_dH_Pp <- Map('%*%', dK_H_K_dH, list(Pp))
				dPe <- Map('+', dK_H_K_dH_Pp, I_K_H_dPp)

				Pe_dPe <- Reduce(rbind, dPe)
				Pe_dPe <- rbind(Pe, Pe_dPe)
			}

			tt <- c(t[j-1], t[j])

			Xp_dXp <- ode(c(Xe_dXe), tt, d_Xp_dXp)

			Xp_dXp <- tail(Xp_dXp, n=1)
			Xp_dXp <- Xp_dXp[-1] # throw away first element (time value)

			Xp <- head(Xp_dXp, n)
			Xp <- as.matrix(Xp)
			dXp <- tail(Xp_dXp, -n)
			dXp <- split(dXp, rep(1:s))

			Pp_dPp <- ode(c(Pe_dPe), tt, d_Pp_dPp)

			Pp_dPp <- tail(Pp_dPp, n=1)
			Pp_dPp <- Pp_dPp[-1] # throw away first element (time value)
			Pp_dPp <- matrix(Pp_dPp, n*(s+1))

			Pp <- head(Pp_dPp, n)
			dPp <- tail(Pp_dPp, -n)
			dPp <- split(dPp, rep(1:s, each=n))
			dPp <- lapply(dPp, function(dPp_i) matrix(dPp_i, n, n))

			# 3
			B <- H %*% Pp %*% t(H) + R
			invB <- inv(B)

			# dB
			dH_Pp_Ht <- Map('%*%', dH, list(Pp %*% t(H)))
			H_dPp_Ht <- lapply(dPp, function(dPp_i) H %*% dPp_i %*% t(H))
			dHt <- Map(t, dH)
			H_Pp_dHt <- Map('%*%', list(H %*% Pp), dHt)

			dB <- Map('+', dH_Pp_Ht, H_dPp_Ht)
			dB <- Map('+', dB, H_Pp_dHt)
			dB <- Map('+', dB, dR)

			# 4
			K <- Pp %*% t(H) %*% invB

			#dK
			dPp_Ht_invB <- Map('%*%', dPp, list(t(H) %*% invB))
			Pp_dHt_invB <- lapply(dH, function(dH_i) Pp %*% t(dH_i) %*% invB)
			Pp_Ht_invB_dB <- Map('%*%', list(Pp %*% t(H) %*% invB), dB)
			Pp_Ht_invB_dB_invB <- Map('%*%', Pp_Ht_invB_dB, list(invB))

			dK <- Map('+', dPp_Ht_invB, Pp_dHt_invB)
			dK <- Map('-', dK, Pp_Ht_invB_dB_invB)

			# 5
			E <- y[, j-1] - H %*% Xp

			# dE
			dH_Xp <- Map('%*%', dH, list(-Xp))
			H_dXp <- Map('%*%', list(-H), dXp)
			dE <- Map('+', dH_Xp, H_dXp)

			# 6
			# Sk
			invB_dB <- Map('%*%', list(invB), dB)
			Et_invB_dE <- Map('%*%', list(t(E) %*% invB), dE)
			Et_invB_dB <- Map('%*%', list(t(E)), invB_dB)
			Et_invB_dB_invB_E <- Map('%*%', Et_invB_dB, list(-.5 * invB %*% E))
			Sp_invB_dB <- Map(Sp, invB_dB)
			Sp_invB_dB <- Map('*', Sp_invB_dB, .5)

			Sk <- Map('+', Et_invB_dE, Et_invB_dB_invB_E)
			Sk <- Map('+', Sk, Sp_invB_dB)

			# 7
			dL <- Map('+', dL, Sk)

			# 8
		}
		return(unlist(dL))
	})
}
