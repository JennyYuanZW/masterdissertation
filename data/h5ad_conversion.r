# =============================================================================
# Description:
#   This script batch-converts Seurat objects saved in .qsave format to AnnData
#   (.h5ad) format. For each .qsave file found in the specified input folder, it:
#     1. Loads the Seurat object using qs::qread()
#     2. Saves it as an H5Seurat file via SeuratDisk::SaveH5Seurat()
#     3. Converts the H5Seurat file to .h5ad using SeuratDisk::Convert()
# =============================================================================

library(qs) # For loading .qsave files
library(Seurat) # For handling Seurat objects
library(SeuratDisk) # For conversion to .h5ad

# Define paths
input_folder <- "/home/zy335/rds/hpc-work/qsave" # Folder with .qsave files
output_folder <- "/home/zy335/rds/hpc-work/h5ad" # Folder to save .h5ad files
log_file <- "/home/zy335/rds/hpc-work/conversion_log.txt" # Log file for errors

# Create output folder if it doesn’t exist
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)

# Get a list of all .qsave files
qsave_files <- list.files(input_folder, pattern = "\\.qsave$", full.names = TRUE)

# Open log file
sink(log_file, append = TRUE)

# Loop through each .qsave file and convert to .h5ad
for (file in qsave_files) {
    try(
        {
            # Extract filename without extension
            file_name <- tools::file_path_sans_ext(basename(file))

            message(paste(Sys.time(), "- Processing:", file))

            # Load Seurat object from .qsave
            seurat_obj <- qs::qread(file)

            # Save as H5Seurat
            h5seurat_path <- file.path(output_folder, paste0(file_name, ".h5Seurat"))
            SaveH5Seurat(seurat_obj, filename = h5seurat_path, overwrite = TRUE)

            # Convert to h5ad
            h5ad_path <- file.path(output_folder, paste0(file_name, ".h5ad"))
            Convert(h5seurat_path, dest = "h5ad", overwrite = TRUE)

            message(paste(Sys.time(), "- Successfully converted:", file, "->", h5ad_path))
        },
        silent = FALSE
    ) # If an error occurs, log it but continue
}

sink()
