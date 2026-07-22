#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    paste(
      "Usage:",
      "Rscript src/Visualization/Vis_FS.R BASELINE_CSV FINETUNED_CSV OUTPUT_PDF [TITLE] [ATTRIBUTE_COL] [METRIC_COL]",
      sep = "\n"
    )
  )
}

baseline_file <- args[[1]]
finetuned_file <- args[[2]]
output_pdf <- args[[3]]
plot_title <- if (length(args) >= 4) args[[4]] else "Fairness Score Comparison"
attribute_col_arg <- if (length(args) >= 5) args[[5]] else NA_character_
metric_col_arg <- if (length(args) >= 6) args[[6]] else NA_character_

pick_col <- function(df, candidates, required_name) {
  nm <- names(df)
  norm <- tolower(gsub("[^a-z0-9]", "", nm))
  cand_norm <- tolower(gsub("[^a-z0-9]", "", candidates))
  hit <- which(norm %in% cand_norm)[1]
  if (is.na(hit)) {
    stop(sprintf("Required column '%s' not found. Available columns: %s", required_name, paste(nm, collapse = ", ")))
  }
  nm[[hit]]
}

read_metric_csv <- function(path, attribute_col, metric_col) {
  df <- read.csv(path, check.names = FALSE)
  group_col <- pick_col(df, c("group"), "group")
  attr_col <- if (!is.na(attribute_col)) attribute_col else pick_col(df, c("occupation", "attribute"), "attribute")
  metric <- if (!is.na(metric_col)) metric_col else pick_col(df, c("BiasScore", "FS"), "BiasScore or FS")

  df %>%
    transmute(
      group = .data[[group_col]],
      attribute = trimws(as.character(.data[[attr_col]])),
      metric = as.numeric(.data[[metric]])
    ) %>%
    filter(!is.na(metric), attribute != "__MEAN__")
}

before_df <- read_metric_csv(baseline_file, attribute_col_arg, metric_col_arg) %>%
  rename(metric_before = metric)

after_df <- read_metric_csv(finetuned_file, attribute_col_arg, metric_col_arg) %>%
  rename(metric_after = metric)

merged_df <- before_df %>%
  inner_join(after_df, by = c("group", "attribute"))

if (nrow(merged_df) == 0) {
  stop("No overlapping rows found between baseline and fine-tuned metric files.")
}

ordered_attributes <- merged_df %>%
  arrange(group, desc(metric_before)) %>%
  pull(attribute) %>%
  unique()

merged_df <- merged_df %>%
  mutate(attribute = factor(attribute, levels = ordered_attributes))

plot_df <- bind_rows(
  merged_df %>% transmute(group, attribute, model = "Baseline", metric = metric_before),
  merged_df %>% transmute(group, attribute, model = "Fine-tuned", metric = metric_after)
)

p <- ggplot(plot_df, aes(x = attribute, y = metric, color = model, shape = model, group = interaction(group, model))) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  facet_wrap(~ group, scales = "free_x") +
  scale_color_manual(values = c("Baseline" = "gray30", "Fine-tuned" = "dodgerblue")) +
  labs(title = plot_title, x = NULL, y = "Fairness Score", color = NULL, shape = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "top"
  )

ggsave(output_pdf, p, width = 12, height = 7, units = "in")
message("Wrote ", output_pdf)
