#####Formatting "Whole_Color_Measurements" 

#Data as output by ImageJ/Fiji needs to be reformatted so each channel is turned into 
#a separate column rather than a row for each image. Final spread sheet should have each 
#image as a row, with each "trait" presented in a separate column for that image (row).

#First set working directory for where your file "Whole_Color_Measurements.csv" is saved
setwd("D:/Macro/Subject B/9")

#Import the data
Whole_Color_Measurements<- read.csv("Whole_color_Measurements.csv", header = T)

#Lets take a look to ensure it has been read into R properly.
#View(Whole_Color_Measurements)

#Add another column denoting what color space and channel the row belongs to
#there is a repeating pattern that should go from the top of the spreadsheet to the bottom
# RGB_R, RGB_G, RGB_B, LAB_L, LAB_A, LAB_B, HSB_H, HSB_S, HSB_B
Colorspace<-c("RGB_R", "RGB_G", "RGB_B", "LAB_L", "LAB_A", "LAB_B", "HSB_H", "HSB_S", "HSB_B")
Whole_Color_Measurements.labeled<-cbind(Whole_Color_Measurements, Colorspace)

#now need to make a column with the "Image.ID" 
#this will be used to denote each row, and columns will be almagamated based on this label.

#check what column image label is in
head(Whole_Color_Measurements.labeled)

#rename column name with "Image.ID"
names(Whole_Color_Measurements.labeled)[2] <- "Image.ID"

#remove ".jpg_HSB", ".jpg_LAB", ".jpg_RGB" from each image.ID so we are just left with the same image ID for each row
Whole_Color_Measurements.labeled$Image.ID <- gsub(".jpg_RGB", "", Whole_Color_Measurements.labeled$Image.ID)
Whole_Color_Measurements.labeled$Image.ID <- gsub(".jpg_LAB", "", Whole_Color_Measurements.labeled$Image.ID)
Whole_Color_Measurements.labeled$Image.ID <- gsub(".jpg_HSB", "", Whole_Color_Measurements.labeled$Image.ID)

head(Whole_Color_Measurements.labeled)

#load necessary package to run next function (dcast)
library(data.table)

###now amalgamate data that is presented in rows into multiple column based on the same row ID
Whole_Color_Measurements.reformat<-dcast(setDT(Whole_Color_Measurements.labeled), Image.ID ~ rowid(Image.ID), value.var = c("Area1","Mean1", "StdDev1", "Mode1", "Min1", "Max1"))

head(Whole_Color_Measurements.reformat)

#rename column headers to denote mean, median, mode, SD of each colorspace+channel
colnames(Whole_Color_Measurements.reformat) <- gsub("_1", "_RGB.R", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_2", "_RGB.G", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_3", "_RGB.B", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_4", "_LAB.L", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_5", "_LAB.A", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_6", "_LAB.B", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_7", "_HSB.H", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_8", "_HSB.S", colnames(Whole_Color_Measurements.reformat))
colnames(Whole_Color_Measurements.reformat) <- gsub("_9", "_HSB.B", colnames(Whole_Color_Measurements.reformat))

head(Whole_Color_Measurements.reformat)

#now save it as a new .csv file
write.csv(Whole_Color_Measurements.reformat, "Whole_Color_Measurements.reformat.csv") 

