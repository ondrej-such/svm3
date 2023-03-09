source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

test3gap <- function(dir = "imagewoof/net1", file = "platt-linear4-1") {
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	load(sprintf("%s/%s.RData", dir, file))

	g1 <- list(list(1,2), list(1,3))
	g2 <- list(list(1,2), list(2,3))
	g3 <- list(list(1,3), list(2,3))

	eval_triple <- function(triple = 3:5) {

        t1 <- triple + 1
		
		iz <- y$target %in% triple
		s_logits <- pair_logits[t1, t1, iz]
        	s_odds <- exp(s_logits)
		N <- sum(iz)

		pred  <- vector("integer", N)
		t2 <- t1 - 1
		pred[y$target[iz] == t2[1]] = 1
		pred[y$target[iz] == t2[2]] = 2
		pred[y$target[iz] == t2[3]] = 3

		p_WLW2<- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
		d_WLW2<- apply(p_WLW2, 1, which.max)
        correct_WLW2 = sum(d_WLW2 == pred)

		p_g1 <- tree_odds(s_odds, g1)
		p_g2 <- tree_odds(s_odds, g2)
		p_g3 <- tree_odds(s_odds, g3)

        d_g = matrix(nrow = N, ncol = 3, 0)
		d_g[,1] <- apply(p_g1, 1, which.max)
		d_g[,2] <- apply(p_g2, 1, which.max)
		d_g[,3] <- apply(p_g3, 1, which.max)

        max_min_graph = max(sapply(1:3, function(i) sum(pred == d_g[,i])))
        avg_min_graph = mean(sapply(1:3, function(i) sum(pred == d_g[,i])))

        hdr <- list(i = t1[1], j = t1[2], k = t1[3], test_cases = N)
        # print(hdr)
        ix = cbind(1:N, pred)
        ll <- function(x) { -mean(log(x[ix]))}
        # print("Before do.call")

	do.call(rbind.data.frame,
            list(
                c(hdr, metric = "Accuracy", model = "WLW2", value = correct_WLW2),
                c(hdr, metric = "Accuracy", model = "mean minimal", value = avg_min_graph),
                c(hdr, metric = "Accuracy", model = "best minimal", value = max_min_graph),
                c(hdr, metric = "Accuracy", model = "improvement", value = correct_WLW2  - max_min_graph),
                c(hdr, metric = "Accuracy", model = "mean improvement", value =correct_WLW2  - avg_min_graph)
                )
            )
	}

	df = data.frame()
	for (i in 0:7) {
		for (j in (i+1):8) {
			for (k in (j+1):9) {
				df <- rbind(df, eval_triple(c(i,j,k)))
			}
		}
	}
   	df$dir = dir
	df$model = factor(df$model,  levels = c(
						"best minimal", "mean minimal", "WLW2", "improvement", "mean improvement"))
	return(df)
}

