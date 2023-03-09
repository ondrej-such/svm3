source("common.R")

fit.svm <- function(dir = "nets/v0", kernel = "linear", nfolds = 4, used_folds = 1, full = F, K = NA, mc.cores =
detectCores()) {
    targetFile = sprintf("%s/svm-%s%d-%d.RData", dir, kernel, nfolds, used_folds)
    if (file.exists(targetFile)) {
        print(sprintf("target %s already exists", targetFile))
        return(NULL)
    }
    file1 <- sprintf("%s/valid_flatten.npy", dir)
    file2 <- sprintf("%s/valid_targets.csv", dir)
    if (!file.exists(file1) || !file.exists(file2)) {
        warning(sprintf("Missing files in directory %s", dir))
        return(NULL)
    }
	x = np$load(file1)
	y <- read_csv(file2)

    if (is.na(K)) {
        K <- max(y$target)
    }

	pairs <- combn(seq.int(0,K), 2)
    pair2 <- combn(seq.int(1, nfolds), used_folds)
	# pairs <- combn(seq.int(0,2), 2)
	stopifnot(nrow(pairs) == 2)

	svm.1 <- function(i) {
		i1 <- y$target == pairs[1,i]
		i2 <- y$target == pairs[2,i]
        wi1 <- which(i1)
        wi2 <- which(i2)

        f1 <- createFolds(wi1, k = nfolds)
        f2 <- createFolds(wi2, k = nfolds)

		t1 <- as.double(Sys.time(), units = "seconds")
        models <- lapply(1:ncol(pair2), function(j) {
            ix1 <- unlist(lapply(1:nrow(pair2), 
                            function(k) wi1[f1[[pair2[k,j] ]]]))
            ix2 <- unlist(lapply(1:nrow(pair2), 
                            function(k) wi2[f2[[pair2[k,j] ]]]))
            ix <- c(ix1, ix2)
            # print(length(ix))
            # print(class(ix))
            # ix = c(wi1[f1[[j]]], wi2[f2[[j]]]) #indices of relevant data for j'th fold
            x1 <- x[ix,]
            y1 <- y$target[ix]
            stopifnot(all(y1 %in% c(pairs[1,i], pairs[2,i])))
            model = svm(x1, as.factor(y1), kernel = kernel, probability = T)
            return(model)
        })

		t2 <- as.double(Sys.time(), units = "seconds")
        print(sprintf("%s %d time = %f", dir, i, t2 - t1))
		# message_parallel("Ending  ", pairs[1,i], " ", pairs[2,i], " duration ", t2 - t1)
		return(list(models = models, time = t2 - t1))
	}

	svm.2 <- function(i) {
		i1 <- y$target == pairs[1,i]
		i2 <- y$target == pairs[2,i]
        wi1 <- which(i1)
        wi2 <- which(i2)

        f1 <- createFolds(wi1, k = nfolds)
        f2 <- createFolds(wi2, k = nfolds)

        pair2 <- combn(seq.int(1, nfolds), used_folds)

		t1 <- as.double(Sys.time(), units = "seconds")
        models = list()
        #models <- lapply(1:ncol(pair2), function(j) {
        for (r in 1:ncol(pair2)) {
            for (s in 1:ncol(pair2)) {
                ix1 <- unlist(lapply(1:nrow(pair2), 
                                function(k) wi1[f1[[pair2[k,r] ]]]))
                ix2 <- unlist(lapply(1:nrow(pair2), 
                                function(k) wi2[f2[[pair2[k,s] ]]]))
                ix <- c(ix1, ix2)
                # ix = c(wi1[f1[[j]]], wi2[f2[[j]]]) #indices of relevant data for j'th fold
                x1 <- x[ix,]
                y1 <- y$target[ix]
                model = svm(x1, as.factor(y1), kernel = kernel, probability = T)
                models <- c(models, model)
            }
        }

		t2 <- as.double(Sys.time(), units = "seconds")
        print(sprintf("%s %d time = %f", dir, i, t2 - t1))
		# message_parallel("Ending  ", pairs[1,i], " ", pairs[2,i], " duration ", t2 - t1)
		return(list(models = models, time = t2 - t1))
	}

    if (full) {
	    results <- mclapply(1:ncol(pairs), svm.2, mc.cores = mc.cores, mc.preschedule = F)
    } else {
	    results <- mclapply(1:ncol(pairs), svm.1, mc.cores = mc.cores, mc.preschedule = F)
    }
	print(sprintf("Net %d done with mclapply in time %f", results$time))

	e <- new.env(parent = emptyenv())
    e[["K"]] = K
    e[["dir"]] = dir
	
	for (i in 1:ncol(pairs)) {
		e[[ sprintf("%d-%d", pairs[1,i], pairs[2,i]) ]] = results[[i]]
	}
	save(e, file = targetFile)
	return(e)
}

