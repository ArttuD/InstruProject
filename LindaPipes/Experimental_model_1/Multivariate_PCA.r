#%%
library(dplyr)
library(robustbase)     # for robust covariance
library(ggplot2)        # for plotting
library(reshape2)       # for reshaping data
#%%
setwd("C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1")
#%%
df <- read.csv("Exp1_N.csv", fileEncoding = "UTF-8")
mapping <- c("collagen 2mg/ml" = 0, "IPN 2.5mM" = 1, "IPN 22mM" = 2, "IPN 40mM" = 3)
df$material_no <- mapping[df$material]
unique(df[, "material_no"])

#%%
selected <- c("cell_line_no","material_no", "area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N")
analysis_df <- subset(df, select = selected)
str(analysis_df)

#%% Pairwise plot
min_max_scale <- function(x) {(x - min(x)) / (max(x) - min(x))}
to_scale <- c("area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N")
scaled_cols <- as.data.frame(lapply(analysis_df[to_scale], min_max_scale))
final <- cbind(analysis_df[c("cell_line_no", "material_no")], scaled_cols)
color_column <- "cell_line_no"  # Column for defining color
color_palette <- c("black", "red", "blue")  # Define a color palette
jpeg("C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/PCA/pairwise_plot.png", 
     width = 800, height = 600, units = "px", quality = 100, bg = "white")
pairs(analysis_df, col = color_palette[analysis_df[, color_column]],
      upper.panel = NULL, gap = 1, cex = 0.5)
dev.off()

#%% Covariance matrix
selected_normal <- c("area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N")
df_cov <- subset(df, select = selected_normal) #removing variables with little variaton 

cov_regular <- cov(df_cov)
cov_regular_rounded <- round(cov_regular, 2)
cov_regular_long <- melt(cov_regular_rounded)
colnames(cov_regular_long) <- c("Variable1", "Variable2", "Covariance")
heatmatplot <-ggplot(cov_regular_long, aes(Variable1, Variable2, fill = Covariance)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, 
                       limit = c(min(cov_regular_long$Covariance), max(cov_regular_long$Covariance)), 
                       name = "Covariance") +
                       geom_text(aes(label = round(Covariance, 2)), color = "black", size = 4)
  theme_minimal() +
  labs(title = "Heatmap of regular covariance Matrix",
       x = "",
       y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/PCA/heatmap_covariance_matrix_regular.png", plot = heatmatplot, width = 8, height = 6, dpi = 300)
write.csv(cov_regular_rounded, "C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/PCA/covariance_matrix_regular.csv", row.names = TRUE)


#%%Robust covariance matrix
# for data that deviate from the normal distribution and contain outliers.
#stimate the multivariate location and scatter using the Minimum Covariance Determinant (MCD) method
selected_robust <- c("area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N")
df_robust_covMcd <- subset(df, select = selected_robust) #removing variables with little variaton 

# Calculating robust covariance matrix
cov_mcd <- robustbase::covMcd(df_robust_covMcd, alpha = 0.7)
# Extracting the matrix
cov_robust <- cov_mcd$cov
cov_robust_rounded <- round(cov_robust, 2)
write.csv(cov_robust_rounded, "C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/PCA/covariance_matrix_robust.csv", row.names = TRUE)

# Convert the covariance matrix to a long format for ggplot
cov_robust_long <- melt(cov_robust_rounded)
colnames(cov_robust_long) <- c("Variable1", "Variable2", "Covariance")
print(cov_robust_long)
heatmatplot_robust <-ggplot(cov_robust_long, aes(Variable1, Variable2, fill = Covariance)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, 
                       limit = c(min(cov_robust_long$Covariance), max(cov_robust_long$Covariance)), 
                       name = "Covariance") +
                       geom_text(aes(label = round(Covariance, 2)), color = "black", size = 4)
  theme_minimal() +
  labs(title = "Heatmap of robust covariance Matrix",
      x = "",
      y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/PCA/heatmap_covariance_matrix_robust.png", plot = heatmatplot_robust, width = 8, height = 6, dpi = 300)

#%%
corrgram(selected_cols, upper.panel = NULL) #correlation matrix as a heatmap
cor_matrix <- cor(analysis_df)
print(round(cor_matrix, 2))
write.table(round(cor_matrix, 2), "C:/Users/srboval1/Time_Series_Instru/TimeSeriesAnalysis/correlation_matrix.csv", sep = ",", quote = FALSE)


class(selected_cols)

cov_regular #Sample covariance
cov_mcd$cov #MCD scatter estimate
cov_mcd$cov - cov_regular #difference between the two

