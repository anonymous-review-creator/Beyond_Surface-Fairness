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
      "Rscript src/Visualization/Vis_SHR.R BASELINE_CSV FINETUNED_CSV OUTPUT_PDF [TITLE] [ATTRIBUTE_COL]",
      sep = "\n"
    )
  )
}

baseline_file <- args[[1]]
finetuned_file <- args[[2]]
output_pdf <- args[[3]]
plot_title <- if (length(args) >= 4) args[[4]] else "ln(SHR) Directionality"
attribute_col_arg <- if (length(args) >= 5) args[[5]] else NA_character_

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

read_shr_csv <- function(path, attribute_col) {
  df <- read.csv(path, check.names = FALSE)
  group_col <- pick_col(df, c("group"), "group")
  attr_col <- if (!is.na(attribute_col)) attribute_col else pick_col(df, c("occupation", "attribute"), "attribute")
  shr_col <- pick_col(df, c("SHR"), "SHR")

  df %>%
    transmute(
      group = .data[[group_col]],
      attribute = trimws(as.character(.data[[attr_col]])),
      SHR = as.numeric(.data[[shr_col]])
    ) %>%
    filter(!is.na(SHR), SHR > 0, attribute != "__MEAN__")
}

base <- read_shr_csv(baseline_file, attribute_col_arg) %>%
  mutate(log_shr_base = log(SHR)) %>%
  select(group, attribute, log_shr_base)

ft <- read_shr_csv(finetuned_file, attribute_col_arg) %>%
  mutate(log_shr_ft = log(SHR)) %>%
  select(group, attribute, log_shr_ft)

df <- base %>%
  inner_join(ft, by = c("group", "attribute")) %>%
  group_by(group) %>%
  arrange(log_shr_base, .by_group = TRUE) %>%
  mutate(row_id = row_number()) %>%
  ungroup() %>%
  mutate(
    y = ifelse(group == "M", row_id, -row_id),
    crosses_zero = (log_shr_base < 0 & log_shr_ft > 0) | (log_shr_base > 0 & log_shr_ft < 0),
    moves_toward_zero = abs(log_shr_ft) < abs(log_shr_base),
    movement = ifelse(moves_toward_zero, "Toward fairness", "Away from fairness")
  )

if (nrow(df) == 0) {
  stop("No overlapping positive-SHR rows found between baseline and fine-tuned metric files.")
}

limit <- max(3, quantile(abs(c(df$log_shr_base, df$log_shr_ft)), 0.98, na.rm = TRUE))
df <- df %>%
  mutate(
    log_shr_base_plot = pmax(pmin(log_shr_base, limit), -limit),
    log_shr_ft_plot = pmax(pmin(log_shr_ft, limit), -limit)
  )

p <- ggplot(df) +
  annotate("rect", xmin = -log(1.2), xmax = log(1.2), ymin = -Inf, ymax = Inf, fill = "grey70", alpha = 0.10) +
  geom_hline(yintercept = 0, color = "steelblue", linewidth = 0.8) +
  geom_vline(xintercept = 0, color = "steelblue", linewidth = 0.8) +
  geom_segment(
    aes(x = log_shr_base_plot, xend = log_shr_ft_plot, y = y, yend = y, color = movement),
    linewidth = 1.0,
    lineend = "round"
  ) +
  geom_point(aes(x = log_shr_ft_plot, y = y, fill = crosses_zero), shape = 21, size = 2.4, color = "black") +
  geom_text(aes(x = limit + 0.15, y = y, label = attribute), hjust = 0, size = 2.8) +
  coord_cartesian(xlim = c(-limit, limit), clip = "off") +
  scale_color_manual(values = c("Toward fairness" = "black", "Away from fairness" = "#ff7f0e")) +
  scale_fill_manual(values = c("FALSE" = "white", "TRUE" = "#d62728"), labels = c("No", "Yes")) +
  scale_y_continuous(name = "Group: M above zero, F below zero", breaks = NULL) +
  labs(
    title = plot_title,
    x = "ln(SHR)",
    color = NULL,
    fill = "Crosses zero"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.margin = margin(10, 100, 10, 10),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

ggsave(output_pdf, p, width = 12, height = 8, units = "in")
message("Wrote ", output_pdf)
