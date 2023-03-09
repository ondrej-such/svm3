source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

test4coll <- function(dir = "cifar10/net1", file = "platt4-1", reps = 200) {
	load(sprintf("%s/%s.RData", dir, file))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))

	g1 <- list(list(1,3), list(1,4), list(2,3), list(2,4))

	eval_tuple <- function() {
		t0 <- sample(0:9, 5)

		t1 <- t0 + 1
		
		iz <- y$target %in% t0
		s_logits <- pair_logits[t1, t1, iz]
		s_odds  <- exp(s_logits)
		N <- sum(iz)

		pred  <- vector("integer", N)
		t2 <- t1 - 1

		for (i in 1:4) {
			pred[y$target[iz] == t2[i]] = i
		}

		iz1 <- y$target %in% t0[1:2]
		logodds12_3 <- pair_logits[t1[1], t1[3], iz1] + pair_logits[t1[3], t1[2], iz1]
		logodds12_4 <- pair_logits[t1[1], t1[4], iz1] + pair_logits[t1[4], t1[2], iz1]
		corr1 = cor(logodds12_3, logodds12_4)

		iz2 <- y$target %in% t0[3:4]
		logodds34_1 <- pair_logits[t1[3], t1[1], iz2] + pair_logits[t1[1], t1[4], iz2]
		logodds34_2 <- pair_logits[t1[3], t1[2], iz2] + pair_logits[t1[2], t1[4], iz2]
		corr2 = cor(logodds34_1, logodds34_2)



        	hdr <- list(i = t0[1], j = t0[2], k = t0[3], l = t0[4], test_cases = N)
        	# print(hdr)
        	ix = cbind(1:N, pred)
        	ll <- function(x) { -mean(log(x[ix]))}
        	# print("Before do.call")

		do.call(rbind.data.frame,
            	list(
                	c(hdr, metric = "Correlation", model = "12", value = corr1),
                	c(hdr, metric = "Correlation", model = "34", value = corr2)
                	 #c(hdr, metric = "Accuracy", model = "G3", value = sum(d_g2 == pred)),
                	 #c(hdr, metric = "Accuracy", model = "G2", value = sum(d_g3 == pred)),
                 #c(hdr, metric = "Accuracy", model = "normal", value = sum(d_normal == pred)),
                 #c(hdr, metric = "Accuracy", model = "WLW2", value = sum(d_WLW2 == pred)),
                 #c(hdr, metric = "Accuracy", model = "stratified", value = sum(d_stratified == pred)),
 #
                 #c(hdr, metric = "NLL", model = "G4", value = ll(p_g1)),
                 #c(hdr, metric = "NLL", model = "G3", value = ll(p_g2)),
                 #c(hdr, metric = "NLL", model = "G2", value = ll(p_g3)),
                 #c(hdr, metric = "NLL", model = "normal", value = ll(p_normal)),
                 #c(hdr, metric = "NLL", model = "WLW2", value = ll(p_WLW2)),
                 #c(hdr, metric = "NLL", model = "stratified", value = ll(p_stratified))
                )
            )


	}

	df = data.frame()
	for (i in 1:reps) {
		df <- rbind(df, eval_tuple())
	}
    	df$dir = dir
	# df$model = factor(df$model, levels = c("G4", "G3", "G2", "WLW2", "stratified", "normal"))
	return(df)
}
