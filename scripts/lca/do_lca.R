#' Perform Latent Class Analysis on annotation data
#' @export
do_lca <- function(data=dt2, cols=c(3,4,5,6,7,8)){
        dt.lca <- data[, cols]
        dt.lca <- data.frame(lapply(dt.lca, factor))
        colnames(dt.lca) <- c('dial', 'sociol', 'jarg', 'styl', 'vart', 'perd')

        lca_formula <- with(dt.lca, cbind(dial, sociol, jarg, styl, vart, perd)~1)

        #------ run a sequence of models with 1-10 classes and print out the model with the lowest BIC
        max_II <- -100000
        min_bic <- 100000
        out_models <- list()
        for(i in 2:10){
        lc <- poLCA(lca_formula, dt.lca, nclass=i, maxiter=3000, 
                    tol=1e-5, na.rm=FALSE,  
                    nrep=10, verbose=TRUE, calc.se=TRUE)
        out_models[[length(out_models)+1]] <- lc
        if(lc$bic < min_bic){
            min_bic <- lc$bic
            LCA_best_model<-lc
        }
    }  
    list(models = out_models, best=LCA_best_model)
}