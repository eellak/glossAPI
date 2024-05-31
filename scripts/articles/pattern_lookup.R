# script for the "Daskalissa" blog post
load("data/gutenberg.rda")
load("data/dimodis.rda")

pattern_lookup <- function(collection, pattern){
	require(stringr)

	pattern_occurence <- sapply(collection, function(x) str_match_all(x$c, pattern=pattern))

	occurence_index <- lapply(seq_along(pattern_occurence), function(ii) which(sapply(pattern_occurence[[ii]], function(x) dim(x)[1]>0)))

	Map(function(x,y) if (length(y)) x[y], pattern_occurence, occurence_index)
}

issa_pattern <- "\\w[[:alpha:]]+ισσα\\w"

dimodis_occurences = pattern_lookup(dimodis$works$ergoes, issa_pattern)
gutenberg_occurences = pattern_lookup(gutenberg$works$ergoes, issa_pattern)

all_occurences <- c(dimodis_occurences, gutenberg_occurences)

uniq_occurences = unique(tolower(unlist(all_occurences)))
print(uniq_occurences)
