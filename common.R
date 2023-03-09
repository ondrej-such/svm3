library(reticulate)
library(e1071)
library(parallel)
library(caret)
library(readr)
source('decode.R')
library(tidyr)
library(dplyr)

TOTAL_NETS = 20

DIVIDE_N = 4

message_parallel <- function(...) {
	system(sprintf('echo "%s\n"', paste0(..., collapse = "")))
}

np <- import("numpy")

prep_dist <- function(dir = "cifar10/net1", file = "platt4-1", train = F) {
    file <- sprintf("%s.RData", file)
   
    if (train) {
        print(sprintf("Using train data %s/%s", dir, file))
        outfile <- sprintf("%s/dist_train.RData", dir)
        stopifnot(file.exists(outfile))
        load(outfile)
        return(dist_matrix)
    }
    outfile <- sprintf("%s/dist_matrix.RData", dir)
    if (file.exists(outfile)) {
        load(outfile)
        return(dist_matrix)
    }
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	print(dim(y))
	load(sprintf("%s/%s", dir, file))
	print(dim(pair_logits))
    K = max(y$target)
    K1 = K + 1

	DM <- matrix(0, nrow = K1, ncol = K1)
	for (i in 1:K) {
		# print(i)
		ii <- y$target == (i - 1)
		# print(sprintf("sum %d", sum(ii)))
		# LM <- matrix(0, nrow = sum(ix), ncol = 10)
		for (j in (i+1):K1) {
			ij <- y$target == (j - 1)
			x1 <- pair_logits[i ,j , ii]
			# print(dim(x1))
			x2 <- pair_logits[i ,j , ij]
			d = abs(mean(x1) - mean(x2)) / sqrt(var(x1) + var(x2))
			DM[i ,j ] = d
			DM[j ,i ] = d

		}
	}
	dist_matrix <- DM
	save(dist_matrix, file = sprintf("%s/dist_test.RData", dir))
	return(DM)
}

summary_fn <- function(x) {
    return(sprintf("%.1f (%.1f)", mean(x), sd(x)))
}

dataset_df <- function(dir = "cifar10/net1", seed = 2022,  cores = detectCores(), test_fn = NA, ...)  {
    RNGkind("L'Ecuyer-CMRG")
    set.seed(seed)
    dirs <- list.files(dir, pattern = "net\\d{1,2}")
    print(dirs)
    dfs <- mclapply(dirs, function(x) test_fn(dir = sprintf("%s/%s", dir, x), ...), mc.cores = cores, mc.set.seed = T)
    big_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    # big_df$dir = dir
    # print(tapply(sum1, c(as.factor(sum1$metric), as.factor(sum1$model)), function(x) x))
    return(big_df)
}


dataset_summary <- function(dir = "cifar10", cores = 1, test_fn = NA, agg_fn = summary_fn, ...)  {
    RNGkind("L'Ecuyer-CMRG")
    dirs <- list.files(dir, pattern = "net\\d{1,2}")
    print(dirs)
    dfs <- mclapply(dirs, function(x) test_fn(dir = sprintf("%s/%s", dir, x), ...), mc.cores = cores, mc.set.seed = T)
    big_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    d1 <- big_df %>% group_by(metric, model) %>% summarise(agg = agg_fn(value))
    m1 <- tapply(d1$agg, list(d1$model, d1$metric), function(x) x)
    print(m1)
    # print(tapply(sum1, c(as.factor(sum1$metric), as.factor(sum1$model)), function(x) x))
    return(list(df = d1, mat = m1))
}

dataset_accuracy <- function(dir = "cifar10", cores = 1, test_fn = test3, ...)  {
    RNGkind("L'Ecuyer-CMRG")
    dirs <- list.files(dir, pattern = "net\\d{1,2}")
    # test.1 <- np$load(sprintf("%s/%s/test_logits.npy", dir, dirs[[1]]))
    # test_cases <- nrow(test.1)
    # print(sprintf("test_cases = %d", test_cases))
    print(dirs)
    dfs <- mclapply(dirs, function(x) test_fn(dir = sprintf("%s/%s", dir, x), ...), 
		    mc.cores = cores, 
		    mc.set.seed = T)
    big_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    # head(big_df)
    test_cases <- big_df$test_cases[1]
    summary_err <- function(x) {
        # print(test_cases)
        return(sprintf("%.1f", 100 * (1 - mean(x) / test_cases)))
    }
    d1 <- big_df %>% 
        filter(metric == "Accuracy") %>% 
        group_by(model) %>% 
        summarise(agg = summary_err(value))
    # m1 <- tapply(d1$agg, list(d1$model, d1$metric), function(x) x)
    # print(m1)
    # print(tapply(sum1, c(as.factor(sum1$metric), as.factor(sum1$model)), function(x) x))
    # return(list(df = d1, mat = m1))
    d1$dir = dir
    return(list(results = big_df, summary = d1))
}

softmax <- function(par){
      n.par <- length(par)
      par1 <- sort(par, decreasing = TRUE)
      Lk <- par1[1]
      for (k in 1:(n.par-1)) {
        Lk <- max(par1[k+1], Lk) + log1p(exp(-abs(par1[k+1] - Lk))) 
      }
      val <- exp(par - Lk)
      return(val)
}

dataset_nll <- function(dir = "cifar10", cores = 1, test_fn = testfull, ...)  {
    RNGkind("L'Ecuyer-CMRG")
    dirs <- list.files(dir, pattern = "net\\d{1,2}")
    test.1 <- np$load(sprintf("%s/%s/test_logits.npy", dir, dirs[[1]]))
    print(dirs)
    dfs <- mclapply(dirs, function(x) test_fn(dir = sprintf("%s/%s", dir, x), ...), mc.cores = cores, mc.set.seed = T)
    big_df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    summary_nll <- function(x) {
        return(sprintf("%.2f", mean(x) ))
    }
    d1 <- big_df %>% 
        filter(metric == "NLL") %>% 
        group_by(model) %>% 
        summarise(agg = summary_nll(value))

    d1$dir = dir

    return(list(results = big_df, summary = d1))
}

datasets_summary <- function(fn = dataset_accuracy, ...) {
    df <- data.frame(fn(dir = "cifar10", ...)$summary)
    print(df)
        for (ds in list("imagenette", "imagewoof")) {
        for (arch in list("resnet18", "resnet34")) {
    for (kind in list("whole", "fresh", "adapt")) {
            df <- rbind(df, fn(dir = sprintf("%s/%s/%s", kind, ds, arch), ...)$summary)
            print(df)
        }

    }

    }
    print(df %>% pivot_wider( names_from = "model", values_from = "agg"))
    return(df)
}

datasets_df <- function(...) {
    df <- data.frame()
    for (ds in list("imagenette", "imagewoof", "cifar10")) {
        for (kernel in list("radial", "linear")) {
            print(c(ds, kernel))
            rdf <-dataset_df(dir = ds, file = sprintf("platt-%s4-1", kernel), ...)
            rdf$dataset = ds
            rdf$kernel = kernel
            df <- rbind(df, rdf)
        }
    }
    # print(df %>% pivot_wider( names_from = "model", values_from = "agg"))
    return(df)
}
