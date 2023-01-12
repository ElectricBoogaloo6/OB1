\name{PSC2.DWDS}
\alias{PSC2.DWDS}
\encoding{UTF-8}
\docType{data}
\title{PSC2.DWDS}
\description{ Potsdam Sentence Corpus 2 with word statistics obtained from the DWDS ("Digitales Wörterbuch der Deutschen Sprache", literally: Digital Dictionary of the German Language), a computerized dictionary of German that is based on a very large text basis and balanced by historical decade (spanning the 20th century) and by text type (fine literature, newspaper articles, professional literature, technical literature and handbooks, transcripts of spoken language). 

NOTE: Usage is limited to private and academic purposes by the following agreement:\cr
\url{http://www.dwds.de/pages/pages_textba/loginnutzung.html}

Word predictabilities were independently collected empirically using a web-based version of the incremental cloze task. For each word in the corpus, a minimum N of 57 participants (mean N: 70) guessed the upcoming word n, given the sentence context up to word n-1. Predictability is defined as the relative frequency of correct guesses.

}
\usage{data(PSC2.DWDS)}
\format{
  A data frame with 1230 observations (words) on the following 16 variables.
  \describe{
\item{\code{sn}}{sentence number (or sentence-ID, Factor w/ 144 levels) } 
\item{\code{nw}}{number of words in sentence}
\item{\code{wn}}{word number in current sentence}
\item{\code{nlett1}}{number of letters, including punctuation ( identical to nchar(PSC2.DWDS[,"word"]) )}
\item{\code{nlett2}}{number of letters, excluding punctuation}
\item{\code{nchr}}{number of characters including preceding space}
\item{\code{l}}{inverse of nchr, used in some regression models (1/nchr)}
\item{\code{wb}}{word boundary in letters, relative to beginning at 0--can be used to compute letter position, given that a fixed width font is used and the offset of the first letter is known}
\item{\code{freq}}{absolute word frequency in DWDS corpus}
\item{\code{f}}{ ??? log10 frequency, with special treatment for extremely rare cases ??? }
\item{\code{bifr}}{frequency of initial bigrams}
\item{\code{trifr}}{frequency of initial trigrams}
\item{\code{pred}}{predictability (by incremental cloze procedure; contains many zeros)}
\item{\code{npred}}{number of participants contributing to predictability score}
\item{\code{cwfw}}{content word or function word (factor with 2 levels)}
\item{\code{word}}{word presented, including subsequent punctuation (character string)}
  }
}
% \details{}
% \source{}
% \references{}
\examples{
data(PSC2.DWDS)
str(PSC2.DWDS)
by(PSC2.DWDS[,c("nlett1","pred","f")], PSC2.DWDS$cwfw, mean)

# overview
par(mfrow=c(2,2))
with(PSC2.DWDS, { 
	cnts<-hist(freq, xlab = 'frequency')
	hist(pred, xlab = 'predictability')
	hist(nlett1, xlab = 'length [letters]')
	plot(f, nlett1, xlab = 'log10 frequency', ylab='length [letters]') 
})

# function word likelihood in sentence beginning, middle, end
ixa = PSC2.DWDS$wn==1               # first word in sentence
ixo = PSC2.DWDS$wn==PSC2.DWDS$nw    # last word in sentence
ixm = !(ixa | ixo)                  # words in between 
for (ix in list(ixa, ixm, ixo)) {
	print(sum(PSC2.DWDS[ix,"cwfw"]=="fw")/sum(ix))
}


}
\keyword{datasets}
