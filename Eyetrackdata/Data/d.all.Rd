\name{d.all}
\alias{d.all}
\docType{data}
\title{ Lesen III, PSC2: Summary statistics}
\description{
  Summary statistics on lesen-III data; one row per combination of word and reader.
}
\usage{data(d.all)}
\format{
  A data frame with 4920 observations on the following 29 variables.
  \describe{
    \item{\code{id}}{subject id}
    \item{\code{sn}}{sentence number}
    \item{\code{wn}}{word number}
    \item{\code{nw}}{number of words in sentence}
    \item{\code{nlett1}}{number of letters (including punctuation)}
    \item{\code{f}}{log10 frequqency}
    \item{\code{bifr}}{initial bigram frequency}
    \item{\code{trifr}}{initial trigram frequency}
    \item{\code{pred}}{cloze predictability}
    \item{\code{npred}}{number of observations contributing to predictability score}
    \item{\code{cwfw}}{content or function word? a factor with levels \code{cw} \code{fw}}
    \item{\code{word}}{a character vector}
    \item{\code{FFDr}}{first fixation duration (standard definition)}
    \item{\code{FFDp}}{first fixation duration (first of several, i.e., without single fixations)}
    \item{\code{SFD}}{single fixation duration}
    \item{\code{GZD}}{gaze duration}
    \item{\code{SKP}}{skipping (0: no, 1: yes)}
    \item{\code{nfp}}{number of first pass fixations}
    \item{\code{nap}}{total number of fixations in any pass}
    \item{\code{TVT}}{total viewing time}
    \item{\code{nsp}}{number of second pass fixations}
    \item{\code{ISL}}{incoming saccade length [letters]}
    \item{\code{OSL}}{outgoing saccade length [letters]}
    \item{\code{ILP}}{initial landing position (relative to word length)}
    \item{\code{s1}}{incoming saccade skipped a word}
    \item{\code{s2}}{outgoing saccade will skip a word}
    \item{\code{haveFirstPass}}{word has been fixated during first pass}
    \item{\code{haveFixation}}{word has been fixated at all}
    \item{\code{ageGroup}}{age group [here 0 because only young adults contribute to the sample]}
  }
}
\details{
  data are a sample from a large cross-sectional lifespan age-comparative study of reading using the Potsdam Sentence Corpus. In the original study, 180 participants between 15 and 80 years of age were tracked reading three different kinds of reading material (PSC, PSC2, PPC).
}
\source{
  Kliegl \& Laubrock (in preparation, working title). Age effects in eye movements during reading.
}
\references{
 Kliegl \& Laubrock (in preparation, working title). Age effects in eye movements during reading.
}
\examples{
data(d.all)
## maybe str(d.all) ; plot(d.all) ...
}
\keyword{datasets}
