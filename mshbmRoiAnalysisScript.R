library(lme4)
library(lmerTest)
library(emmeans)
library(dplyr)

file <- "/Users/caroline/Desktop/Dissertation/TI_Backchannels/ROIanalysis/mshbm_TI_BI_runlevel_surface_betas.csv"

dat <- read.csv(file)
head(dat)
str(dat)

dat$ROI <- recode(
  dat$ROI,
  "DU_DN_A"  = "DN-A",
  "DU_DN_B"  = "DN-B",
  "DU_FPN_B" = "FPN-B"
)


dat$Subject <- factor(dat$Subject)
dat$Run <- factor(dat$Run)

# BI as reference condition
dat$Condition <- factor(
  dat$Condition,
  levels = c("BI", "TI")
)

# DN-A as reference ROI
dat$ROI <- factor(
  dat$ROI,
  levels = c("DN-A", "DN-B", "FPN-B")
)


cat("\nNumber of subjects:\n")
print(length(unique(dat$Subject)))

cat("\nNumber of observations:\n")
print(nrow(dat))

cat("\nObservations by ROI and condition:\n")
print(
  table(dat$ROI, dat$Condition)
)

cat("\nObservations by run and condition:\n")
print(
  table(dat$Run, dat$Condition)
)

cat("\nMissing beta estimates:\n")
print(sum(is.na(dat$Beta)))


cat("\nDESCRIPTIVES\n")

descriptives <- dat %>%
  group_by(ROI, Condition) %>%
  summarise(
    N = n(),
    Mean = mean(Beta, na.rm = TRUE),
    SD = sd(Beta, na.rm = TRUE),
    .groups = "drop"
  )

print(descriptives)


model <- lmer(
  Beta ~ Condition * ROI + (1 | Subject),
  data = dat,
  REML = TRUE
)


cat("\nMODEL SUMMARY\n")

print(summary(model))

emm <- emmeans(
  model,
  ~ Condition | ROI
)

cat("\nESTIMATED MARGINAL MEANS\n")

print(
  emm,
  infer = c(TRUE, FALSE)
)

bi_ti <- contrast(
  emm,
  method = list(
    "BI - TI" = c(1, -1)
  )
)

cat("\nBI VS TI WITHIN EACH ROI\n")

print(
  bi_ti,
  infer = c(TRUE, TRUE),
  adjust = "none"
)

cat("\n95% CONFIDENCE INTERVALS\n")

print(
  confint(
    bi_ti,
    level = 0.95,
    adjust = "none"
  )
)