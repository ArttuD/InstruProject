#%%
library(dplyr)
df <- read.csv('C:/Users/srboval1/OneDrive - Aalto University/Instru/Datafiles/Exp1/Exp1_N.csv') 
numerical_df <- df[, sapply(df, is.numeric)]

print(head(numerical_df))
#%%
analysis_df <- subset(df, select = c("cell_line", "material", "area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N"))
min_max_scale <- function(x) {(x - min(x)) / (max(x) - min(x))}
to_scale <- c("area_N", "convexity_perimeter_N", "compactness_N", "aspect_ratio_N")
scaled_cols <- as.data.frame(lapply(analysis_df[to_scale], min_max_scale))
final <- cbind(analysis_df[c("cell_line", "material")], scaled_cols)

color_column <- "cell_line"  # Column for defining color
color_palette <- c("black", "red", "blue")  # Define a color palette

jpeg("C:/Users/srboval1/pairwise_plot.jpg", 
     width = 800, height = 600, units = "px", quality = 100, bg = "white")

# Create the pairwise plot using pairs() function
pairs(analysis_df, 
      col = color_palette[analysis_df[, color_column]],
      upper.panel = NULL, gap = 1, cex= 0.85)

# Close the JPEG device
dev.off()

