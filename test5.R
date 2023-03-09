source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

test5 <- function(dir = "cifar10/net1", file = "platt4-1", reps = 200) {
	load(sprintf("%s/%s.RData", dir, file))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))

	g1 <- list(list(1,2), list(2,3), list(3,4), list(4,5))
	g2 <- list(list(1,2), list(2,3), list(3,4), list(2,5))
	g3 <- list(list(1,2), list(1,3), list(1,4), list(1,5))


	eval_tuple <- function(run_id) {

		t0 <- sample(0:9, 5)

		t1 <- t0 + 1
		
		iz <- y$target %in% t0
		s_logits <- pair_logits[t1, t1, iz]
		s_odds  <- exp(s_logits)
		N <- sum(iz)

		pred  <- vector("integer", N)
		t2 <- t1 - 1

		for (i in 1:5) {
			pred[y$target[iz] == t2[i]] = i
		}


		p_WLW2 <- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
		d_WLW2 <- apply(p_WLW2, 1, which.max)


		p_g1 <- tree_odds(s_odds, g1)
		p_g2 <- tree_odds(s_odds, g2)
		p_g3 <- tree_odds(s_odds, g3)

		d_g1 <- apply(p_g1, 1, which.max)
		d_g2 <- apply(p_g2, 1, which.max)
		d_g3 <- apply(p_g3, 1, which.max)

        hdr <- list(i = t0[1], j = t0[2], k = t0[3], l = t0[4], m = t0[5], test_cases = N, run_id = run_id)
        # print(hdr)
        ix = cbind(1:N, pred)
        ll <- function(x) { -mean(log(x[ix]))}
        # print("Before do.call")

		do.call(rbind.data.frame,
            list(
                c(hdr, metric = "Accuracy", model = "G4", value = sum(d_g1 == pred)),
                c(hdr, metric = "Accuracy", model = "G3", value = sum(d_g2 == pred)),
                c(hdr, metric = "Accuracy", model = "G2", value = sum(d_g3 == pred)),
                c(hdr, metric = "Accuracy", model = "WLW2", value = sum(d_WLW2 == pred))
                )
            )
	}

	df = data.frame()
	for (i in 1:reps) {
		df <- rbind(df, eval_tuple(i))
	}
    	df$dir = dir
	df$model = factor(df$model, levels = c("G4", "G3", "G2", "WLW2"))
	return(df)
}
