source("common.R")
# library(mclust)
library(stringr)
library(readr)

prep_platt_probs <- function(dir = "cifar10/net", file = "svm-linear4-1", kind = "test", mc.cores = detectCores(), agg =
mean, verbose = F, perClass = 200) {
    print(sprintf("In prep_platt_probs %d", as.integer(verbose)))

    load(sprintf("%s/%s.RData", dir, file))
    K <- e[["K"]]
    K1 <- K + 1

    if (kind != "subset") {
    	x = np$load(sprintf("%s/%s_flatten.npy", dir, kind))
    	y <- read.csv(sprintf("%s/%s_targets.csv", dir, kind))
    } else {
    	x1 = np$load(sprintf("%s/%s_flatten.npy", dir, "valid"))
    	y1 <- read_csv(sprintf("%s/%s_targets.csv", dir, "valid"))

	tf <- sprintf("%s/subset_targets.csv", dir)
	if (file.exists(tf)) {
		df.target <- read_csv(tf)
	} else {
		if (is.na(perClass)) {
    			y1 <- read.csv(sprintf("%s/%s_targets.csv", dir, "test"))
			perClass = y1 %/% K1
		}
		df.target = data.frame()
		for (i in 0:K) {
			ix = sample(which(y1$target == i), perClass)
			dfn <- data.frame(idx = ix, target = i)
			df.target <- rbind(df.target, dfn)
		}
		write_csv(df.target, tf)
	}
	x <- x1[df.target$idx, ]
	y <- y1[df.target$idx,]
    }


    c2 <- combn(0:K, 2)
    print("Before parallel computation")

    par.r <- mclapply(1:ncol(c2), function(k)  {
        i = c2[1,k]
        j = c2[2,k]
        stopifnot(i < j)
        mn <- sprintf("%d-%d", i, j)
        models <- e[[mn]]$models
        nmod <- length(models)
        logits <- matrix(0, nrow = length(y$target), ncol = nmod)

	if(verbose) {
		print(sprintf("Doing pair %d-%d", i, j))
    	}


        for (m in 1:nmod) {
            mod = models[[m]]
            preds <- predict(mod, x, decision.values = T)
            dvs <- attr(preds,"decision.values")


            stopifnot(mod$levels[1] == toString(min(i,j)))
            stopifnot(mod$levels[2] == toString(max(i,j)))
            stopifnot(length(colnames(dvs)) == 1)

            if (mod$levels[mod$labels[1] ] == i) {
                stopifnot(sprintf("%d/%d", i, j) ==
                      colnames(mod$decision.names)[1])
                stopifnot(sprintf("%d/%d", i, j) ==
                      colnames(dvs)[1])
                logits[,m] <-  - (dvs * mod$probA + mod$probB)

            } else {
                stopifnot(sprintf("%d/%d", j, i) ==
                      colnames(mod$decision.names)[1])
                stopifnot(sprintf("%d/%d", j, i) ==
                      colnames(dvs)[1])

                    logits[,m] <- (dvs * mod$probA + mod$probB)
            }
        }
        logits_agg <- apply(logits, 1, agg)
        logits_median <- apply(logits, 1, median)
        ix <- y$target %in% c(i,j)
        c_agg <- sum((y$target[ix] == i) == (logits_agg[ix] > 0))
        c_median <- sum((y$target[ix] == i) == (logits_median[ix] > 0))
        # print(sprintf("%s %d/%d", dir, i, K))
        return(list(i = i, j = j, logits = logits_agg, acc1 = c_agg / sum(ix), acc_median = c_median / sum(ix)))
 	},
        mc.cores = mc.cores, 
        mc.preschedule = TRUE
    )
    print("Done with parallel portion")

    vm <- array(0, dim = c(K1, K1, nrow(y)))

    for (r in par.r) {
        i = r$i
        j = r$j
        # print(sprintf("Collecting %d,%d", i, j))
        vm[i + 1, j + 1, ] = r$logits
        vm[j + 1, i + 1, ] = -r$logits
    }
    print(sprintf("Mean agg accuracy is %f", mean(sapply(par.r, function(x) {x$acc1}))))
    print(sprintf("Mean median accuracy is %f", mean(sapply(par.r, function(x) x$acc_median))))
    return(vm)
}

prep_platt <- function(dir = "cifar10/net1", file = "svm-linear4-1", mc.cores = detectCores(), theta = Inf, agg = mean,
...) {
    vm <- prep_platt_probs(dir = dir, file = file, mc.cores = mc.cores, agg = agg, kind = "test", ...) 
    vm = pmax(vm, -theta)
    vm = pmin(vm, theta)
    print(dim(vm))

    pair_logits = vm

    sm <- prep_platt_probs(dir = dir, file = file, mc.cores = mc.cores, agg = agg, kind = "subset", ...)
    sm = pmax(sm, -theta)
    sm = pmin(sm, theta)
    subset_logits = sm
    ofile <- sprintf("%s/%s.RData", dir, str_replace(file, "svm", "platt"))
    save(pair_logits, subset_logits, file = ofile)

    extra <- sprintf("%s/valid_from_train_flatten.npy", dir) 
    if (file.exists(extra)) {
        em <- prep_platt_probs(dir = dir, file = file, mc.cores = mc.cores, agg = agg, kind = "valid_from_train", ...)

        em = pmax(em, -theta)
        em = pmin(em, theta)
        extra_logits = em
        save(extra_logits, file = sprintf("%s/%s.RData", dir, str_replace(file, "svm", "extra"))) 
    }
}

prep_gmm <- function(dir = "nets/v0", mc.cores = 10, theta = Inf) {
    tdx <- np$load(sprintf("%s/valid_flatten.npy", dir))
    tdy <- read.csv(sprintf("%s/valid_targets.csv", dir))
	x = np$load(sprintf("%s/test_flatten.npy", dir))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	load(sprintf("%s/mod.RData", dir))

    K <- e[["K"]]
    K1 <- K + 1

    c2 <- combn(0:K, 2)

    par.r <- mclapply(1:ncol(c2), function(k)  {
            i = c2[1,k]
            j = c2[2,k]
            stopifnot(i < j)
            mn <- sprintf("%d-%d", i, j)
            models <- e[[mn]]$models
            stopifnot(length(models) == DIVIDE_N)
            logits <- vector("double", length(y$target))

            for (mod in models) {
                predd <- predict(mod, tdx, decision.values = T)
                dvsd <- attr(predd, "decision.values")

                mod.i <- densityMclust(dvsd[tdy$target == i], verbose = F)
                mod.j <- densityMclust(dvsd[tdy$target == j], verbose = F)

                preds <- predict(mod, x, decision.values = T)
                dvs <- attr(preds,"decision.values")

                logits <- logits + (predict(mod.i, dvs, what = "dens", logarithm = T) 
                                -predict(mod.j, dvs, what = "dens", logirithm = T))
            }
            logits <- logits / DIVIDE_N
            print(sprintf("%s %d:%d/%d", dir, i, j, K))
            return(list(i = i, j = j, logits = logits))
	    },
        mc.cores = mc.cores, 
        mc.preschedule = TRUE
    )
    print("Done with parallel portion")

	vm <- array(0, dim = c(K1, K1, nrow(y)))

    for (r in par.r) {
        i = r$i
        j = r$j
        # print(sprintf("Collecting %d,%d", i, j))
        vm[i + 1, j + 1, ] = r$logits
        vm[j + 1, i + 1, ] = -r$logits
    }
    print(dim(vm))
    vm = pmax(vm, -theta)
    vm = pmin(vm, theta)
    print(dim(vm))

	pair_logits = vm
    save(pair_logits, file = sprintf("%s/gmm.RData", dir ))
}

prep_logis <- function(dir = "nets/v0", file = "logmod4-1", mc.cores = 1, theta = Inf, agg = mean) {
	x = np$load(sprintf("%s/test_flatten.npy", dir))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	load(sprintf("%s/%s.RData", dir, file))

    K <- e[["K"]]
    K1 <- K + 1

    c2 <- combn(0:K, 2)

    par.r <- mclapply(1:ncol(c2), function(k)  {
            i = c2[1,k]
            j = c2[2,k]
            stopifnot(i < j)
            mn <- sprintf("%d-%d", i, j)
            model <- e[[mn]]$model
            # nmod <- length(models)
            #klogits <- matrix(0, nrow = length(y$target), ncol = nmod)

            # for (m in 1:nmod) {
            logits <- predict.glm(mod, x)
                 #dvs <- attr(preds,"decision.values")

            ix <- y$target %in% c(i,j)
            c_corr <- sum((y$target[ix] == i) == (logits[ix] > 0))
            # print(sprintf("%s %d/%d", dir, i, K))
            return(list(i = i, j = j, logits = logits, acc1 = c_corr / sum(ix)))
	    },
        mc.cores = mc.cores, 
        mc.preschedule = TRUE
    )
    print("Done with parallel portion")

	vm <- array(0, dim = c(K1, K1, nrow(y)))

    for (r in par.r) {
        i = r$i
        j = r$j
        # print(sprintf("Collecting %d,%d", i, j))
        vm[i + 1, j + 1, ] = r$logits
        vm[j + 1, i + 1, ] = -r$logits
    }
    print(sprintf("Mean accuracy is %f", mean(sapply(par.r, function(x) {x$acc1}))))
    vm = pmax(vm, -theta)
    vm = pmin(vm, theta)
    print(dim(vm))

	pair_logits = vm

    ofile <- sprintf("%s/%s.RData", dir, str_replace(file, "mod", "platt"))
    save(pair_logits, file = ofile)
}
