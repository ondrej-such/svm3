source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

test4add <- function(dir = "cifar10/net1", file = "platt-radial4-1") {
	load(sprintf("%s/%s.RData", dir, file))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
    ys <- read_csv(sprintf("%s/subset_targets.csv", dir))

	eval_triple <- function(triple = 0:2) {
	    df = list()


        sort_triple <- function(triple, k) {
            t0 <- triple
		    t1 <- t0 + 1
		    t2 <- t1 - 1
            t3 <- c(t2, k)

            g <- list(list(1,4), list(2,4), list(3,4))
		    iz <- ys$target %in% c(triple, k)
		    N <- sum(iz)
		    pred  <- vector("integer", N)
		    for (i in 1:4) {
			    pred[ys$target[iz] == t3[i]] = i
		    }

            t4 <- c(t1, k1)
		    m_logits <- subset_logits[t4, t4, iz]
            m_odds <- exp(m_logits)

            p_g <- tree_odds(m_odds, g)
            d_g <- apply(p_g, 1, which.max)

            M <- matrix(0, nrow = 4, ncol = 4)
            for (i in 1:N) {
                M[pred[i], d_g[i]] = M[pred[i], d_g[i]] + 1
            }
            M <- M + t(M)

            t <- t1
            d0 <- c(M[2, 3], M[1, 3], M[1, 2])
            o <-order(d0)
            t1 <- t[o]
            # stopifnot(M[t1[2], t1[3]] <= M[t1[1], t1[3]])
            # stopifnot(M[t1[1], t1[3]] <= M[t1[1], t1[2]])
            return(list(triple = t1))
        }


        class10 <- 1:10
        t0 <- triple + 1
        # print(triple)

        for (k1 in class10[-t0]) {
            k = k1 - 1
            
		    # t2 <- t1 - 1
            t1 <- sort_triple(triple, k)$triple 
            t2 <- t1 - 1
            t3 <- c(t2, k)
            t4 <- c(t1, k1)
		
		    iz <- y$target %in% c(triple, k)
		    s_logits <- pair_logits[t4, t4, iz]
		    N <- sum(iz)
		    pred  <- vector("integer", N)

		    for (i in 1:4) {
			    pred[y$target[iz] == t3[i]] = i
		    }
      	    hdr <- list(i = t0[1], j = t0[2], k = t0[3], l = k1, test_cases = N)
            p_wu <- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
            d_wu <- apply(p_wu, 1, which.max)
            corr_full <- sum(d_wu == pred)

            for (i in 1:2) {
                for (j in (i + 1):3) {
                    s_logits[i,j,] = pair_logits[t1[i],k1, iz] + pair_logits[k1, t1[j], iz]
                    s_logits[j,i,] = -s_logits[i,j,]
                }
            }
            
            p_wu <- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
            d_wu <- apply(p_wu, 1, which.max)
            df <- append(df, list(c(hdr, metric = "Accuracy", model = "D0", 
                value = sum(d_wu == pred))))

            for (i in 1:3) {
                i_logits = s_logits
                edge <- t1[-i]
                stopifnot(length(edge) == 2)
                k = edge[1]
                l = edge[2]
                e2 = (1:3)[-i]
                ki = e2[1]
                li = e2[2]
                i_logits[ki, li,] = pair_logits[k, l, iz]
                i_logits[li, ki,] = -i_logits[ki,li,]
            
                p_wu <- t(sapply(1:N, function(i) wu2_ld(i_logits[,,i])))
                d_wu <- apply(p_wu, 1, which.max)

        	    ix = cbind(1:N, pred)
        	    ll <- function(x) { -mean(log(x[ix]))}

                df <- append(df, list(c(hdr, metric = "Accuracy", model = sprintf("D%d", i),
                value = sum(d_wu == pred))))

            } # loop i 

            df <- append(df, list(c(hdr, metric = "Accuracy", model = "DF", 
                value = corr_full)))
	    } # loop k
        df <- do.call(rbind.data.frame, c(df, stringsAsFactors = F))
        # print(tail(df))
        return(df)
    }
    

	df2 = data.frame(stringsAsFactors = F)
    for (i in 0:7) {
        for (j in (i+1):8) {
            for (k in (j+1):9) {
                df2 <- rbind(df2, eval_triple(c(i,j,k)))
            }
        }
    }
    df2$dir = dir
	return(df2)
}
