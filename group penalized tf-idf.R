# custom_tfidf.R
# Author: Ben Amini (2025)
# Group-penalized TF-IDF weighting for think-aloud / cognitive transcript analysis.

library(quanteda)
library(textstem)
library(Matrix)


data1 <- read.csv("path")
data2 <- read.csv("path")
data <-rbind(data1, data2)
View(custom_tfidf(data$Utterance, data$Question))
View(data)


custom_tfidf <- function(
    texts,
    item_id,
    custom_stopwords = NULL,
    min_termfreq = 1
) {
  # Create corpus from text vector
  corp <- quanteda::corpus(texts)
  
  # Tokenize
  toks <- quanteda::tokens(
    corp,
    remove_punct = TRUE,
    remove_numbers = TRUE
  )
  
  # Lemmatize
  toks <- quanteda::as.tokens(lapply(as.list(toks), textstem::lemmatize_words))
  
  # Remove stopwords if provided
  if (!is.null(custom_stopwords)) {
    toks <- quanteda::tokens_remove(toks, custom_stopwords, padding = FALSE)
  }
  
  # Create DFM
  dfm <- quanteda::dfm(toks)
  
 
  quanteda::docvars(dfm, "item") <- item_id
  
  # Trim infrequent terms
  if (min_termfreq > 1) {
    dfm <- quanteda::dfm_trim(dfm, min_termfreq = min_termfreq)
  }
  
  # Group penalty
  dfm_grouped <- quanteda::dfm_group(dfm, groups = quanteda::docvars(dfm, "item"))
  G_t <- Matrix::colSums(dfm_grouped > 0)
  G   <- quanteda::ndoc(dfm_grouped)
  
  penalty <- log((G_t + 1) / (G + 1))
  penalty <- penalty - min(penalty) + 0.01
  
  dfm_pen <- quanteda::dfm_weight(dfm, weights = penalty)
  
  # Global IDF
  N   <- quanteda::ndoc(dfm_pen)
  DF  <- quanteda::docfreq(dfm_pen)
  idf <- log(N / (DF + 1))
  
  # Final TF * Penalty * IDF
  tfidf_dfm <- quanteda::dfm_weight(dfm_pen, weights = idf)
  
  # Drop zero-variance features
  m <- as.matrix(tfidf_dfm)
  keep <- which(apply(m, 2, function(x) var(x) > 0))
  tfidf_dfm <- tfidf_dfm[, keep]
  
  return(as(tfidf_dfm, "dgCMatrix"))
}

