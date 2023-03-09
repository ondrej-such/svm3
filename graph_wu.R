source('common.R')
library(Rcpp)
sourceCpp("decode.cpp")

graph_wu <- function(dir = "cifar10/net1", 
                       file = "platt4-1", 
                       graph = "graphs/K5-5", 
                       model = "mean", 
                       mc.cores = detectCores(),
                       reps = 50) {
	load(sprintf("%s/%s.RData", dir, file))
	y <- read.csv(sprintf("%s/test_targets.csv", dir))
	nn <- np$load(sprintf("%s/test_logits.npy", dir))

    edges <- matrix(as.integer(as.matrix(read_csv(graph))), ncol = 2)
    edges <- edges - min(edges) + 1
    # print(edges)
    # print(dim(edges))
    # print(class(edges[1,1]))

    K1 = max(y$target)
    K = K1 + 1

    t0 <- 0:K1
    t1 <- 1:K
    iz <- y$target %in% t0
    s_logits <- pair_logits[t1, t1, iz]
    N <- sum(iz)

    pred  <- vector("integer", N)
    t2 <- t1 - 1

    for (i in 1:K) {
        pred[y$target[iz] == t2[i]] = i
    }

    vertices = base::unique(c(edges[,1], edges[,2]))
    # print(vertices)
    E = matrix(0, nrow = nrow(edges), ncol = 2)

	eval_tuple <- function(run_id) {
        perm <- sample(vertices, K)
        M <- matrix(F, nrow = K, ncol = K)
        for (i in 1:nrow(edges)) {
            E[i,1] = perm[edges[i,1]]
            E[i,2] = perm[edges[i,2]]
            M[E[i,1], E[i,2]] = T
            M[E[i,2], E[i,1]] = T
        }
        mdeg = max(apply(M,1, sum))
        buffer = matrix(0, nrow = N, ncol = mdeg)
        s_logits = array(0, dim = c(K, K, N))
        for (i in 1:(K - 1)) {
            for (j in (i + 1):K) {
                if(M[i,j]) {
                    s_logits[i,j,] = pair_logits[i,j,]
                    s_logits[j,i,] = pair_logits[j,i,]
                    next
                }
                col = 0
                for (m in 1:K) {
                    if (! M[i, m])
                        next
                    if (!M [m,j])
                        next
                    col = col + 1
                    buffer[,col] = pair_logits[i,m,] + pair_logits[m,j,]
                }
                stopifnot(col > 0)
                if (col == 1) {
                    r = buffer[,1]
                } else  {
                    if (model == "mean") {
                        r = apply(buffer[,1:col], 1, mean)
                    } else {
                        r = apply(buffer[,1:col], 1, median)
                    }
                }
                s_logits[i,j,] = r
                s_logits[j,i,] = -r
            }
        }

		p_wu <- t(sapply(1:N, function(i) wu2_ld(s_logits[,,i])))
        # print(p_normal)
		d_wu <- apply(p_wu, 1, which.max)


        hdr <- list(test_cases = N, n_edges = nrow(edges), model = basename(graph), run_id = run_id)
        # print(hdr)
        ix = cbind(1:N, pred)
        ll <- function(x) { -mean(log(x[iz]))}
        # print("Before do.call")

		do.call(rbind.data.frame,
            list(
                c(hdr, metric = "Accuracy", value = sum(d_wu == pred)),
                c(hdr, metric = "NLL", value = ll(p_wu))
                )
            )
	}

	dfs <- mclapply(1:reps, function(i) {
        print(i)
        eval_tuple(i)
        # row$rep = i
        }, mc.cores = mc.cores
        )
	df <- do.call(rbind.data.frame, c(dfs, stringsAsFactors = F))
    df$dir = dir
    df$graph = graph
    df1 <- df %>% filter(metric == "Accuracy")
    df2 <- df %>% filter(metric == "NLL")
    print(fivenum(df1$value))
    print(fivenum(df2$value))

	return(df)
}

graphs_data <- function(dir = "cifar10/net1", 
                       graphs = "graphs", 
                       model = "mean", 
                       reps = 50) {
    ofile = sprintf("%s/graphs.RData", dir)
    if (file.exists(ofile)) {
        load(ofile)
        return(df.graph)
    }
    df <- data.frame()
    for (kernel in c("linear", "radial")) {
	    y <- read.csv(sprintf("%s/test_targets.csv", dir))


        for (graph in list.files(graphs, full.name = T)) {
            edges <- matrix(as.integer(as.matrix(read_csv(graph))), ncol = 2)
            edges <- edges - min(edges) 
            vertices = base::unique(c(edges[,1], edges[,2]))
            if (max(vertices) == max(y$target)) {
                print(sprintf("Doing graph %s", graph))
                dfn <- graph_wu(dir = dir, 
                                file = sprintf("platt-%s4-1", kernel),
                                graph = graph, 
                                model = model, 
                                reps = reps)
                dfn$kernel = kernel
                df <- rbind(df, dfn)
            }
        }
    }
    df.graph <- df
    save(df.graph, file = ofile)
    return(df.graph)
}

graphs10 <- function() {
    df <- data.frame()
    for (ds in c("cifar10", "imagewoof", "imagenette")) {
        for (gf in list.files(ds, pattern = "graphs.RData", full.name = T, recursive = T)) {
            load(gf)
            df.graph$dataset = ds
            df <- rbind(df, df.graph)
        }
    }
    df.graph10 <- df
    save(df.graph10, file = "data.frames/graph10.RData")
}
