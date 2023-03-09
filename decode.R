
# graph has to be a spanning tree
#
# graph example arg: list(list(1,2), list(1,3))
tree_odds <- function(odds, graph) {
	e = new.env(parent = emptyenv())
	e[[toString(graph[[1]][[1]]) ]] = rep(1, dim(odds)[3])

	while(length(graph) > 0) {
		for (i in 1:length(graph)) {
			v1 <- graph[[i]][[1]]
			v2 <- graph[[i]][[2]]
			s1 <- toString(v1)
			s2 <- toString(v2)
			nex1 <- is.null(e[[s1]])
	                nex2 <- is.null(e[[s2]])
			if (nex1 == nex2) 
				next
			if (nex1) {
				e[[s1]] = e[[s2]] * odds[v1, v2, ]
				
			} else {

				e[[s2]] = e[[s1]] * odds[v2, v1, ]
			}
			graph[[i]] <- NULL
			break
		}
	}
	M1 <- matrix(0, ncol = dim(odds)[[1]], nrow = dim(odds)[[3]])

	for (i in 1:dim(odds)[[1]]) {
		M1[,i] <- e[[toString(i)]]
	}
	M2 <- M1 * (1 / apply(M1, 1, sum))
	return(M2)
}
