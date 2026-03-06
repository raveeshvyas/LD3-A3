library(tidyverse)

data <- read.csv("file.csv") 

ggplot(data, aes(x = RTlexdec)) +
  geom_histogram(bins = 30, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of RTlexdec")

ggplot(data, aes(x = RTlexdec)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "black", alpha = 0.7) +
  facet_wrap(~AgeSubject) + 
  theme_minimal() +
  labs(title = "RT Distribution by Subject Age")


young_data <- data %>% filter(tolower(AgeSubject) == "young") 

young_data <- young_data %>% 
  mutate(z_score = (RTlexdec - mean(RTlexdec, na.rm = TRUE)) / sd(RTlexdec, na.rm = TRUE))

pct_gt_196 <- mean(young_data$z_score > 1.96, na.rm = TRUE) * 100
pct_lt_neg196 <- mean(young_data$z_score < -1.96, na.rm = TRUE) * 100
pct_gt_3 <- mean(young_data$z_score > 3, na.rm = TRUE) * 100

cat("Percentage of data with z-score > 1.96:", pct_gt_196, "%\n")
cat("Percentage of data with z-score < -1.96:", pct_lt_neg196, "%\n")
cat("Percentage of data with z-score > 3:", pct_gt_3, "%\n")

outlier_words <- young_data %>% filter(z_score > 3) %>% select(Word, RTlexdec, z_score)
print(head(outlier_words))

cat("RTlexdec - Mean:", mean(young_data$RTlexdec, na.rm=TRUE), "Median:", median(young_data$RTlexdec, na.rm=TRUE), "\n")
cat("NounFrequency - Mean:", mean(young_data$NounFrequency, na.rm=TRUE), "Median:", median(young_data$NounFrequency, na.rm=TRUE), "\n")

young_data <- young_data %>% 
  mutate(starts_with_p = ifelse(startsWith(tolower(Word), "p"), "p_word", "other"))

t_test_p <- t.test(RTlexdec ~ starts_with_p, data = young_data)
print(t_test_p)

young_nouns_verbs <- young_data %>% 
  filter(WordCategory %in% c("N", "V", "Noun", "Verb"))

ggplot(young_nouns_verbs, aes(x = WordCategory, y = RTlexdec)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Noun RTs vs Verb RTs")

print("Five number summary for Nouns:")
fivenum(young_data$RTlexdec[young_data$WordCategory %in% c("N", "Noun")]) 

summary_stats <- young_nouns_verbs %>%
  group_by(WordCategory) %>%
  summarise(
    mean_RT = mean(RTlexdec, na.rm = TRUE),
    sd_RT = sd(RTlexdec, na.rm = TRUE),
    n = n()
  ) %>%
  mutate(
    se = sd_RT / sqrt(n),
    ci_lower = mean_RT - 1.96 * se,
    ci_upper = mean_RT + 1.96 * se
  )

ggplot(summary_stats, aes(x = WordCategory, y = mean_RT, fill = WordCategory)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
  theme_minimal() +
  labs(title = "Mean RT by Word Category with 95% CI")

young_data <- young_data %>% mutate(first_letter = substr(tolower(Word), 1, 1))
ggplot(young_data, aes(x = first_letter, y = RTlexdec)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "RTs by Initial Letter")

vowels <- c("a", "e", "i", "o", "u")
young_data <- young_data %>%
  mutate(
    first_char = substr(tolower(Word), 1, 1),
    second_char = substr(tolower(Word), 2, 2),
    two_consonants = ifelse(!(first_char %in% vowels) & !(second_char %in% vowels), "Yes", "No")
  )

t_test_cons <- t.test(RTlexdec ~ two_consonants, data = young_data)
print(t_test_cons)