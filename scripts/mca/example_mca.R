library(ca)
dt2 <- read.csv("annot_data_raw.dat", sep="\t", stringsAsFactors=F)
dt3 <- na.omit(dt2)
mjca1 <- mjca(dt2[dt2$Ποικιλία %in% c("κνε", "καθαρεύουσα") & !(dt2$Ύφος %in% c("τραγικό", "λυρικό", "κωμικό", "παραινετικό")),], )
plot(mjca1)
# coordinates
mjca1$rowcoord
