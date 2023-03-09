source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

test6 <- function(dir = "nets/v0", file = "platt4-1", reps = 200) {
	load(sprintf("%s/%s.RData", dir, file))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	nn <- np$load(sprintf("%s/test_logits.npy", dir))



	eval_tuple <- function(run_id) {

        if (!(run_id %% 20)) {
            print(run_id)
        }

		t0 <- sample(0:9, 6)

		t1 <- t0 + 1
		
		iz <- y$target %in% t0
		N <- sum(iz)

		pred  <- vector("integer", N)
		t2 <- t1 - 1

		for (i in 1:6) {
			pred[y$target[iz] == t2[i]] = i
		}


        


        hdr <- list(i = t0[1], j = t0[2], k = t0[3], l = t0[4], m = t0[5], n = t0[6], test_cases = N, run_id = run_id)
        ix = cbind(1:N, pred)
        ll <- function(x) { -mean(log(x[ix]))}


        fit.6 <- function(method = "mean") {

            fit.3 <- function(v1, v2, vs) {
                m <- sapply(vs, function(v) pair_logits[t1[v1], t1[v], iz] + pair_logits[t1[v], t1[v2], iz])
                # print(dim(m))
                stopifnot(ncol(m) == 3)
                p <- if (method == "mean") {
                    apply(m, 1, mean)    
                } else if (method == "median") {
                    apply(m, 1, median)
                }
            }

            n_logits <- array(0, dim = c(6,6, N))
            for (i in 1:3) {
                for (j in 4:6) {
                    n_logits[i,j,] = pair_logits[t1[i], t1[j], iz]
                }
            }

            n_logits[1,2,] = fit.3(1,2, 4:6) 
            n_logits[1,3,] = fit.3(1,3, 4:6) 
            n_logits[2,3,] = fit.3(2,3, 4:6) 

            n_logits[4,5,] = fit.3(4,5, 1:3) 
            n_logits[4,6,] = fit.3(4,6, 1:3) 
            n_logits[5,6,] = fit.3(5,6, 1:3) 

            for (i in 1:5) {
                for (j in (i+1):6) {
                    n_logits[j,i,] = -n_logits[i,j,]
                }
            }

		    p_wu <- t(sapply(1:N, function(i) wu2_ld(n_logits[,,i])))
		    d_wu <- apply(p_wu, 1, which.max)
            return(list(c(hdr, metric = "Accuracy", model = method, value = sum(d_wu == pred)),
                        c(hdr, metric = "NLL", model = method, value = ll(p_wu))
                        ))
        }

        # print(hdr)
        # print("Before do.call")
        s_logits = array(0, dim = c(6,6, N))
        for (i in 1:6) {
            for (j in 1:6) {
                s_logits[i,j,] = pair_logits[t1[i], t1[j], iz]
            }
        }
        p_wu <- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
        d_wu <- apply(p_wu, 1, which.max)
        lf <- list(c(hdr, metric = "Accuracy", model = "full", value = sum(d_wu == pred)),
                        c(hdr, metric = "NLL", model = "full", value = ll(p_wu)))

		do.call(rbind.data.frame,
            c( fit.6("median"),
               fit.6("mean"),
               lf
            )
          )


	}

	df = data.frame()
	for (i in 1:reps) {
		df <- rbind(df, eval_tuple(i))
	}
    df$dir = dir
	return(df)
}



