source("common.R")

source_python("py_complexity.py")

time.fit.svm <- function(x, y,
        nfolds = 4,
        split = "ova",
        method = "libsvm",
        used_folds = 1,
        full = F,
        K = max(y$target),
        minN = 8,
        maxN = max(9, as.integer(floor(K * 0.43))),
        kernel = "radial") {

    stopifnot(min(y$target) == 0)

    N <- sample.int(maxN - minN + 1, size = 1) + minN - 1 
    stopifnot(N > 1)
    classes <- sample.int(K + 1, size = N) - 1

    if (split == "ova")  {
        class1 <- classes[1]
        class2 <- classes[-1]
    } else if (split == "ecoc") {
        N2 = as.integer(floor(N / 2))
        N = 2 * N2
        class1 = classes[1:N2]
        class2 = classes[(N2 + 1) : N]
    } else {
        stopifnot(split == "crammer_singer")
        if (method != "liblinear") {stop("Crammer-singer is only supported for method liblinear")}
        class1 = classes
        class2 = c()
    }
    print(sprintf("Starting fit %s, %s, %s", method, split, kernel))
    print(sprintf("K = %d, N = %d in (%d,%d)", K, N, minN, maxN))


	i1 <- y$target %in% class1
	i2 <- y$target %in% class2
    wi1 <- which(i1)
    wi2 <- which(i2)

    f1 <- createFolds(wi1, k = nfolds)
    if (length(wi2) > 0) {
        f2 <- createFolds(wi2, k = nfolds)}

	t1 <- as.double(Sys.time(), units = "seconds")
    ix1 <- unlist(lapply(1:used_folds,
            function(k) wi1[f1[[k ]]]))
    if (length(wi2) > 0) {
        ix2 <- unlist(lapply(1:used_folds,
                function(k) wi2[f2[[k ]]]))
    ix <- c(ix1, ix2)
    } else {
       ix <- ix1
    }
    x1 <- x[ix,]
    stopifnot(all(y$target[ix] %in% classes))
    y1 <- y$target[ix] %in% class1

    if (method == "libsvm") {
        model <- svm(x1, as.factor(y1), kernel = kernel, probability = T)
        correct <- sum(y1 == predict(model, x1)) / length(ix)
        nSV <- nrow(model$SV)
    } else {
        stopifnot(kernel == "linear")
        if (split == "crammer_singer") {
            correct <- fit_liblinear(
                x1, np_array(y$target[ix]), multi_class = "crammer_singer")
        }
        else {
            correct <- fit_liblinear(x1, np_array(as.integer(y1)))
        }
        nSV <- NA
    }


    t2 <- as.double(Sys.time(), units = "seconds")
    print(sprintf("Finished fit %s, %s, %s", method, split, kernel))
    print(sprintf("%f correct", correct))
    print(sprintf("%d time = %f", N, t2 - t1))
		# message_parallel("Ending  ", pairs[1,i], " ", pairs[2,i], " duration ", t2 - t1)
    return(
        list(
            N = N, time = t2 - t1,
            N1 = length(class1), N2 = length(class2),
            split = split, kernel = kernel,
            method = method, correct = correct, SV = nSV))
}


loop_test <- function(x, y, time_hr = 1/60, ...) {
    df <- data.frame()
    t1 <- Sys.time()
    t2 <- t1

    while (as.double(t2 - t1, units = "hours") < time_hr) {
        row <- time.fit.svm(x, y, ...)
        print("Finished fit:")
        df <- rbind(df, row)
        print(tail(df, 1))

        t2 <- Sys.time()
    }
    df$valid = T
    df$valid[nrow(df)] = F
    print(" Omitting -->")
    print(tail(df, 1))
    print(" <--")
    return(df)
}

loop.par.test <- function(
    dir = "cifar10/net1", 
    time_limit_hr = 1/60, 
    cores = detectCores() / 2, ...) {
    print(sprintf("Cores = %d, limit = %f", cores, time_limit_hr))

    file1 <- sprintf("%s/valid_flatten.npy", dir)
    file2 <- sprintf("%s/valid_targets.csv", dir)
    if (!file.exists(file1) || !file.exists(file2)) {
        warning(sprintf("Missing files in directory %s", dir))
        return(NULL)
    }
    x = np$load(file1)
    y <- read_csv(file2)

    if (cores > 1) {
        dfs <- mclapply(1:cores, 
            function(i) loop_test(x = x , y = y , time_hr = time_limit_hr, ...), 
            mc.cores = cores)
        df <- do.call(rbind.data.frame,dfs)
    } else {
        df <- loop_test(x = x, y = y, time_hr = time_limit_hr, ...)
    }
    return(df)
}

eval.complexity <- function(time_limit_hr = 1/60, ...) {
    # df1 <- loop.par.test(dir = "imagenet-50/net1", time_limit_hr, split = "ova", kernel = "linear", minN = 10, ... )
    # save(df1, file = "data.frames/df1.RData")
    if (!file.exists("data.frames/df2.RData")) {
        df2 <- loop.par.test(dir = "imagenet-50/net1", time_limit_hr, split = "ova", kernel = "radial", minN = 10, ... )
        save(df2, file = "data.frames/df2.RData")
    } else {
        load("data.frames/df2.RData")
    }
    
    if (!file.exists("data.frames/df4.RData")) {
        df4 <- loop.par.test(dir = "imagenet-50/net1", time_limit_hr, split = "ecoc", kernel = "radial", minN = 10, ... )
        save(df4, file = "data.frames/df4.RData")
    } else {
        load("data.frames/df4.RData") 
    }
    
    # df3 <- loop.par.test(dir = "imagenet-50/net1", time_limit_hr, split = "ecoc", kernel = "linear", minN = 10, ... )
    # save(df3, file = "data.frames/df3.RData")
    # df.complexity <- rbind(df1, df2, df3, df4)
    df.complexity <- rbind(df2, df4)
    save(df.complexity, file = "data.frames/complexity.RData")
    return(df.complexity)
}

eval.complexity.liblinear <- function(time_limit_hr = 1/60, ...) {
    if (!file.exists("data.frames/complx_liblin_cs.RData")) {
        print("Testing crammer_singer")
        complx_liblin_cs <- loop.par.test(
            dir = "imagenet-50/net1", time_limit_hr = time_limit_hr,
            split = "crammer_singer", kernel = "linear",
            method = "liblinear", ...)
        save(complx_liblin_cs, file = "data.frames/complx_liblin_cs.RData")
    } else {
        load("data.frames/complx_liblin_cs.RData")
    }

    if (!file.exists("data.frames/complx_liblin_ova.RData")) {
        print("Testing ova")
        complx_liblin_ova <- loop.par.test(
            dir = "imagenet-50/net1", time_limit_hr = time_limit_hr,
            split = "ova", kernel = "linear",
            method = "liblinear", ...)
        save(complx_liblin_ova, file = "data.frames/complx_liblin_ova.RData")
    } else {
        load("data.frames/complx_liblin_ova.RData")
    }

    if (!file.exists("data.frames/complx_liblin_ecoc.RData")) {
        print("Testing ecoc")
        complx_liblin_ecoc <- loop.par.test(
            dir = "imagenet-50/net1", time_limit_hr = time_limit_hr,
            split = "ecoc", kernel = "linear",
            method = "liblinear", ...)
        save(complx_liblin_ecoc, file = "data.frames/complx_liblin_ecoc.RData")
    } else {
        load("data.frames/complx_liblin_ecoc.RData")
    }

    df.complx.liblin <- rbind(complx_liblin_ova, complx_liblin_ecoc, complx_liblin_cs)
    save(df.complx.liblin, file = "data.frames/complexity.liblin.RData")
    return(df.complx.liblin)
}

eval.complexity.liblinear(time_limit_hr = 12, minN = 6, maxN = 20)