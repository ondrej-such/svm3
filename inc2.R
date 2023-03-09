
source('common.R')
library(ape)
library(Rcpp)
sourceCpp("decode.cpp")
library(stringr)

confusion_matrix <- function(dir)  {
    fname <- sprintf("%s/conf_train.RData", dir)
    if (file.exists(fname)) {
    	load(fname)
    	return(conf_matrix)
    }
    y1 <- read.csv(sprintf("%s/valid_targets.csv", dir))
    y2 <- read.csv(sprintf("%s/test_targets.csv", dir))
    nn1 <- np$load(sprintf("%s/valid_logits.npy", dir))
    nn2 <- np$load(sprintf("%s/test_logits.npy", dir))
    
    K = max(y1$target)
    K1 = K + 1
    t0 <- 0:K

    t1 <- t0 + 1
    iz1 <- y1$target %in% t0
    iz2 <- y2$target %in% t0

    t2 <- t1 - 1

    Md <- matrix(0, nrow = K1, ncol = K1)
    
    d_nn1 <- apply(nn1, 1, which.max)
    d_nn2 <- apply(nn2, 1, which.max)
    N1 <- sum(iz1)
    N2 <- sum(iz2)

    pred1  <- vector("integer", N1)
    pred2  <- vector("integer", N2)

    for (i in 1:K1) {
        pred1[y1$target[iz1] == t2[i]] = i
        pred2[y2$target[iz2] == t2[i]] = i
    }

    for (i in 1:N1) {
        # Minimal spanning tree requires negative errors
        #
        Md[pred1[i], d_nn1[i]] = 1 + Md[pred1[i], d_nn1[i]]
    }
    conf_matrix  = Md
    save(conf_matrix, file = fname)

    return(Md)
}

inc <- function(dir = "cifar10/net1", 
		star_center = NA,  # if specified, used the given class as the star center
       	# random_init = T,   # use random initialization for star center
        file = "platt-linear4-1", # file containing pair odds in logits format
        verbose = F,       # spew debug info during computation 
        theta = Inf,        # bound entries in pair logits matrix 
        model = "max1-edge"
		) 
{
    plattFile <- sprintf("%s/%s.RData", dir, file)
    print(plattFile)
    stopifnot(file.exists(plattFile))
    load(plattFile)
    stopifnot(theta > 0)
    p1 <- pmax(pair_logits, -theta)
    p2 <- pmin(p1, theta)
    pair_logits <- p2

    # Mdnew <- (Md + t(Md)) / 2
    # Md <- Mdnew


    y1 <- read.csv(sprintf("%s/valid_targets.csv", dir))
    y2 <- read.csv(sprintf("%s/test_targets.csv", dir))
    nn1 <- np$load(sprintf("%s/valid_logits.npy", dir))
    nn2 <- np$load(sprintf("%s/test_logits.npy", dir))

    # print("inc: reading files 5 ")
    K = max(y1$target)
    K1 = K + 1
    print(sprintf("Number of classes is %d", K1))

    Mb <- matrix(T, nrow = K1, ncol = K1) # matrix indicating whether an element is available
    # M <- matrix(0, nrow = K1, ncol = K1) # confusion matrix
    if (model != "edge") {
        vb <- rep(T, K1)
    }
    for (i in 1:K1)  {Mb[i,i] = F} # mark diagonal entries as not available

    t0 <- 0:K

    t1 <- t0 + 1
    iz1 <- y1$target %in% t0
    iz2 <- y2$target %in% t0
    N1 <- sum(iz1)
    N2 <- sum(iz2)
    pred1  <- vector("integer", N1)
    pred2  <- vector("integer", N2)
    t2 <- t1 - 1

    for (i in 1:K1) {
        pred1[y1$target[iz1] == t2[i]] = i
        pred2[y2$target[iz2] == t2[i]] = i
    }
    d_nn1 <- apply(nn1, 1, which.max)
    d_nn2 <- apply(nn2, 1, which.max)

    df <- data.frame()

    a_logits <- array(0, dim = c(K1, K1, N2))
    if (startsWith(model, "max1")) {
        s_logits <- array(0, dim = dim(subset_logits))
        N3 <- dim(s_logits)[3]
        df.target <- read_csv(sprintf("%s/subset_targets.csv", dir))
        pred3 = df.target$target + 1
    }

    if (startsWith(model, "max3") || startsWith(model, "max4")) {
        load(sprintf("%s/%s.RData", dir, str_replace(file, "platt", "extra"))) 
        s_logits <- array(0, dim = dim(extra_logits))
        subset_logits <- extra_logits
        N3 <- dim(s_logits)[3]
        df.target <- read_csv(sprintf("%s/valid_from_train_targets.csv", dir))
        pred3 = df.target$target + 1
    }

    if (!exists("s_logits")) {
        N3 = N2
        subset_logits <- pair_logits
        s_logits = array(0, dim = dim (subset_logits))
    }
    
    # pred3 <- vector("integer", N3)

    p_nn2 <- t(apply(nn2, 1, softmax))
    nn2_ll <- -mean(log(p_nn2[cbind(1:N2, pred2)]))

    est_odds <- function() {
        for (i in 1:K) {
            for (j in (i+1):K1) {
                if(!Mb[i, j])
                    # nothing to do here, odds are exact from SVM
                    next
                tot <- 0
                est <- rep(0., N2)
		        es2 <- rep(0., N3)
                for (k in 1:K1) {
                    if ((!Mb[i,k]) && (!Mb[k, j])) {
                       est = est + (a_logits[i,k, ] + a_logits[k, j, ])
		               es2 = es2 + (s_logits[i,k, ] + s_logits[k, j, ])
                       tot <- tot + 1
                    }
                }
                stopifnot(tot > 0)
                est <- est / tot
                a_logits[i,j,] <<- est
                a_logits[j,i,] <<- -est

		        es2 <- es2 / tot
                s_logits[i,j,] <<- es2
                s_logits[j,i,] <<- -es2
            }
        }
    }

    update_odds <- function(r,c) {
        for (i in 1:K) {
            for (j in (i+1):K1) {
                if(!Mb[i, j])
                    # nothing to do here, odds are exact from SVM
                    next

                if (! ((i %in% c(r,c)) || (j %in% c(r,c))))
                    next

                tot <- 0
                est <- rep(0., N2)
                es2 <- rep(0., N3)
                for (k in 1:K1) {
                    if ((!Mb[i,k]) && (!Mb[k, j])) {
                       est = est + (a_logits[i,k, ] + a_logits[k, j, ])
                       es2 = es2 + (s_logits[i,k, ] + s_logits[k, j, ])
                       tot <- tot + 1
                    }
                }
                stopifnot(tot > 0)
                est <- est / tot
                es2 <- es2 / tot
                a_logits[i,j,] <<- est
                a_logits[j,i,] <<- -est
                s_logits[i,j,] <<- es2
                s_logits[j,i,] <<- -es2
            }
        }
    }

    set_odds <- function(i,j) {
        if (i == j)
            return()
        # iz = y$target %in% (-1 + c(i,j))
        a_logits[i,j,] <<- pair_logits[i,j,]
        a_logits[j,i,] <<- pair_logits[j,i,]
        s_logits[i,j,] <<- subset_logits[i,j,]
        s_logits[j,i,] <<- subset_logits[j,i,]
        Mb[i,j] <<- F
        Mb[j,i] <<- F
        return()
    }


    if (is.na(star_center)) {
        mi = sample(K1, 1)
    } else {
        mi = 1 + ((star_center - 1) %% K1)
    }
    print(sprintf("Chose class %d as the base of the star", mi))
    star_center <- mi
    if (endsWith(model, "star")) {
        vb[mi] = F
    }

    if (mi == 1) {
        g1 <- lapply((mi+1):K1, function(i) list(mi, i))
        for (i in 2:K1) {
            Mb[i, 1] = Mb[1, i ] = F
            set_odds(i,1)
        }
    } else {
        gl <- lapply(1:(mi-1), function(i) list(i, mi))
        gg <- lapply((mi+1):K1, function(i) list(mi, i))
        g1 <- c(gl, gg)

        for (i in 1:K1) {
            if (i == mi)
                next
            Mb[i, mi] = Mb[mi, i] = F
            set_odds(i, mi)
        }
    }
    est_odds()

    step = 1 
    last_error = 1000
    row = NA
    col = NA
    d_nn2 <- apply(nn2, 1, which.max)

    while (TRUE) {
	    p_WLW2 <- t(sapply(1:N2, function(i) wu2_ld(a_logits[,,i])))
	    d_WLW2 <- apply(p_WLW2, 1, which.max)

        M = matrix(0, nrow = K1, ncol = K1) 
        for (i in 1:N2) {
            j = pred2[i]
            k = d_WLW2[i]
            if (j !=k ) {
                M[j,k] = M[j,k] + 1
            }
        }

        if (startsWith(model, "max1") | startsWith(model, "max3") | startsWith(model, "max4")) {
	        p_WLW3 <- t(sapply(1:N3, function(i) wu2_ld(s_logits[,,i])))
	        d_WLW3 <- apply(p_WLW3, 1, which.max)


            M3 = matrix(0, nrow = K1, ncol = K1) 
            for (i in 1:N3) {
                j = pred3[i]
                k = d_WLW3[i]
                if (j != k) {
                    M3[j,k] = M3[j,k] + 1
                }
            }
        }

        if (startsWith(model, "max4")) {
            wlw2 = sum(d_WLW3 == pred3)
            wlw2_ll = -mean(log(p_WLW3[cbind(1:N3, pred3)]))
        } else {
	        wlw2 = sum(d_WLW2 == pred2)
		    wlw2_ll = -mean(log(p_WLW2[cbind(1:N2, pred2)]))
        }

	    row_data <- list(
                star_center = star_center,
                step = step,
                n_edges = (K1 * (K1 - 1) - sum(Mb)) / 2,
                nn = sum(d_nn2 == pred2),
			    WLW2 = wlw2,
			    WLW2_ll = wlw2_ll,
                nn_ll = nn2_ll,
                errors = last_error,
                model = model, 
                file = file, 
                dir = dir,
                row = row,
                col = col
	   )

        if (startsWith(model, "pre-")) {
            mi <- which.max(apply(M, 1, sum))
            df <- inc(dir = dir, 
                star_center = mi, 
                model = sub("pre-", "", model)
                )
            df$n_edges = df$n_edges + K
            df$model = model
            return(df)
        }


        M = M + t(M)
        df <- rbind(df, row_data)
        if(sum(Mb) == 0) {
            break
        }
        if (verbose) {
    	    print(df)
	        print(sum(Mb))
        }

        step <- step + 1
        if (step > 10) {
            # break
        }

        if (endsWith(model, "-edge")) {
            if (startsWith(model, "random")) {
                ix0 <- Mb
            } else {
        	    nzi <- which(Mb, arr.ind = T)
		        if (startsWith(model, "max2") | startsWith(model, "max4")) {
	        	    max_val <- max(M[nzi])
               	    ix0 <- (M == max_val) & Mb
		        } else {
			        stopifnot(startsWith(model, "max1") | startsWith(model, "max3"))
	        	    max_val <- max(M3[nzi])
                	ix0 <- (M3 == max_val) & Mb
		        }
            }
            ix <- which(ix0, arr.ind = T)
            stopifnot(nrow(ix) > 0)
            ixi <- sample(1:nrow(ix), 1)
            row = ix[ixi, 1]
            col = ix[ixi, 2]
            max_val = M[row, col]
            last_error = max_val
    
	    stopifnot(Mb[row,col])
            stopifnot(Mb[col,row])
	    stopifnot(row != col)
    
            set_odds(row, col)
            update_odds(row, col)
        } else {
            stopifnot(endsWith(model, "-star"))
            vi = which(vb)
            stopifnot(all(vb[vi]))
            if (startsWith(model, "random")) {
                i = sample(vi, 1)
                max_val = sum(M[i,])
            } else {
                costs <- sapply(vi, function(v) sum(M[v,which(Mb[v,])]))
                max_val = max(costs)
                ix0 <- (max_val == costs)
                ix <- sample(which(ix0), 1)
                i <- vi[ix]
            }
            last_error = max_val

            for (j in 1:K1) {
                if (Mb[i,j]) {
                    set_odds(i, j)
                    update_odds(i,j)
                }
            }
            # star_center <- i
            row <- i
            col <- i 
            stopifnot(vb[i])
            vb[i] <- F
        }
        if (step %% 20 == 0) {
            print(tail(df))
        }
	}
    stopifnot(all(a_logits == pair_logits))
    stopifnot(all(s_logits == subset_logits))
    return(df)
}

prep_matrix <- function(desc, dir) {
	if (desc == "uniform") {
        fname <- sprintf("%s/dist_matrix.RData", dir)
        stopifnot(file.exists(fname))
        load(fname)
        K1 = nrow(dist_matrix)
		return(matrix(1, nrow = K1, ncol = K1))
	}
	if (desc == "confusion") {
		return(-confusion_matrix(dir))
	}
	if (desc == "dist_train") {
		return(prep_dist(dir = dir, train = T))
	}
	if (desc == "dist_test") {
		return(prep_dist(dir = dir, train = F))
	}
	stopifnot(0)
}

inc_all <- function(dir = "imagewoof/net1",
                verbose = F,
                runs = 50,
                mc.cores = detectCores(), ...) {
    for (kernel in c("linear", "radial")) {
        # inc_data(model = "max-star", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "max1-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "max2-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "random-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores, ...)
        inc_data(model = "random-star", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores, ...)
    }
}

inc_EV <- function(dir = "imagenet-50/extra_valid",
                verbose = F,
                runs = 50,
                mc.cores = detectCores(), ...) {
    for (kernel in c("linear", "radial")) {
        # inc_data(model = "max-star", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "max1-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "max2-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores,  ...)
        inc_data(model = "max3-edge", dir = dir, kernel = kernel, verbose = verbose, runs = runs, mc.cores = mc.cores, ...)
    }
}



inc_one <- function(star_center = 8, runs = 20, mc.cores = 10, dir = "../examples/imagenet/net50", ...) {
    RNGkind("L'Ecuyer-CMRG")
    dfs <- mclapply(1:runs, function(i) {
        df <- inc(dir = dir, Md = prep_matrix("uniform", dir), star_center = star_center, ...)
        df$run_id = i
        print(sprintf("Run %d finished", i))
        df
    }, mc.cores = mc.cores, mc.set.seed = T)
    ofile <- sprintf("%s/inc-%d.RData", dir, star_center)
    one_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    save(one_df, file = ofile)
}

inc_data <- function(
		     dir = "../examples/imagenet/net50", 
             model = "random-star",
		     verbose = F,
             replace = F,
             seed = 2022,
             runs = 10, 
             kernel = "linear", 
             mc.cores = detectCores(), ...) {
    f = sprintf("platt-%s4-1", kernel)
    print(sprintf("Doing %s matrix", model))
    RNGkind("L'Ecuyer-CMRG")
    set.seed(seed)
    ofile <- sprintf("%s/inc2.RData", dir)
    base_id <- if (!replace & file.exists(ofile)) {
        load(ofile)
	    if (any(inc_df$model == model)) {
        	max(inc_df$run_id[inc_df$model == model])
	    } else 0
    } else 0
    t1 <- Sys.time()
    dfs <- mclapply(1:runs, function(i) {
        df <- inc(dir = dir, file = f, verbose = verbose, model = model, star_center = i + base_id, ...)
        df$kernel = kernel
        df$run_id = i + base_id
        print(sprintf("Run %d out of %d is done", i, runs))
        df
    },  mc.cores = mc.cores, mc.set.seed = T)
    t2 <- Sys.time()

    new_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    new_df$model = model
    inc_df <- if (file.exists(ofile)) {
        load(ofile)
        old_df <- 
        if (replace) {
            inc_df[inc_df$model != model, ] 
        } else inc_df
        rbind(old_df, new_df)
    }  else new_df
    save(inc_df, file = ofile)
    dt <- as.numeric(difftime(time1 = t2, time2 = t1, units = "mins"))
    print(sprintf("Multicore computation took %f minutes", dt))
}

inc2 <- function(runs = 50, ...) {
    for (i in 1:20) {
        inc_all(dir = sprintf("imagewoof/net%d", i), runs = runs, ...)
        inc_all(dir = sprintf("imagenette/net%d", i), runs = runs, ...)
        inc_all(dir = sprintf("cifar10/net%d", i), runs = runs, ...)
    }
    inc_all("imagenet-50/net1", runs = runs, ...)
    load("imagenet-50/net1/inc2.RData")
    df <- inc_df
    for (i in 1:20) {
        for (ds in c("imagewoof", "imagenette", "cifar10")) {
            load(sprintf("%s/net%d/inc2.RData", ds, i))
            df <- rbind(df, inc_df)
        }
    }
    return(df)
}

inc10 <- function(...) {
    df <- data.frame()
    for (i in 1:20) {
        for (ds in c("imagewoof", "imagenette", "cifar10")) {
            if (!file.exists(sprintf("%s/net%d/inc2.RData", ds, i))) { 
                inc_all(dir = sprintf("%s/net%d", ds, i), ...)
            }
            load(sprintf("%s/net%d/inc2.RData", ds, i))
            df <- rbind(df, inc_df)
        }
    }
    return(df)
}

inc50 <- function(...) {
    df <- data.frame()
    dir <- "imagenet-50/net1"
    dfile <- sprintf("%s/inc2.RData", dir)
    if (!file.exists(dfile)) { 
         inc_all(dir = dir, ...)
    }
    load(dfile)
    return(inc_df)
}

inc3 <- function(dir = "imagenet-50/extra_valid", ...) {
    for (kernel in c("linear", "radial")) {
        pfile <- sprintf("%s/platt-%s4-1.RData", dir, kernel)
        if (!file.exists(pfile)) {
            print(sprintf("Skipping %s kernel", kernel))
            next
        }
        print(sprintf("Doing %s kernel", kernel))
        inc_data(dir = dir, model = "max1-edge", kernel = kernel, ...)
        inc_data(dir = dir, model = "max2-edge", kernel = kernel, ...)
        inc_data(dir = dir, model = "max3-edge", kernel = kernel, ...)
    }
}

inc_eva <- function(...) {
    df <- data.frame()
    for (i in 1:20) {
        if (!file.exists(sprintf("imagenette/eva%d/inc2.RData", i))) { 
            inc3(dir = sprintf("imagenette/eva%d", i), ...)
        }
        load(sprintf("imagenette/eva%d/inc2.RData", i))
        df <- rbind(df, inc_df)
    }
    return(df)
}

inc_dataset <- function(dir = "adapt/imagewoof/resnet18", label = "imagewoof") {
    fl1 <- list.files(path = dir, "inc.RData", full.names = T, recursive = T)
    stopifnot(length(fl1) > 0)
    df <- data.frame()
    for (f in fl1) {
        print(f)
        load(f)
        df <- rbind(df, inc_df)
    }
    df$dataset = label
    of <- sprintf("../%s.RData", label)
    print(sprintf("Saving to file %s", of))
    save(df, file = of)
    return(df)
}

