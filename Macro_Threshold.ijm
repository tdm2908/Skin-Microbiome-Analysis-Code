// User selects the working directory
input = getDirectory("Choose Source Directory ");
list = getFileList(input);

// Loop through each file in the directory
for (i = 0; i < list.length; i++) {
    Wholecolor(input, list[i]);
}

function Wholecolor(input, filename) {
    // Open the image
    open(input + filename);
    
    // Crop the image
    makeOval(550, 350, 1850, 1850);
    setBackgroundColor(0, 0, 0);
    run("Clear Outside");

    // Determine min[2] value
    getHistogram(values, counts, 256);
    var maxCountIndex = 0;
    var maxCount = counts[0];
    for (var i = 1; i < counts.length; i++) {
        if (counts[i] > 15 && counts[i] > maxCount) {
            maxCount = counts[i];
            maxCountIndex = i;
        }
    }
    // Find the corresponding value and add 20
    var maxValue = values[maxCountIndex];
    var min_value = maxValue + 85;
    
    // Open ROI manager
    run("ROI Manager...");
    roiManager("Show All");

    // Assign raw opened image the name "currentImage"
    currentImage = getImageID();

    // Duplicate raw image to make a copy that will be modified to make a mask
    run("Duplicate...", "title=Mask");

    // Highlight the "Mask" image as the duplicate that will be thresholded to make the mask
    selectWindow("Mask");
    
    // Color Threshold "Mask" image to select only the object of interest
    min = newArray(3);
    max = newArray(3);
    filter = newArray(3);
    a = getTitle();
    run("HSB Stack");
    run("Convert Stack to Images");
    selectWindow("Hue");
    rename("0");
    selectWindow("Saturation");
    rename("1");
    selectWindow("Brightness");
    rename("2");
    min[0] = 0;
    max[0] = 255;
    filter[0] = "pass";
    min[1] = 0;
    max[1] = 255;
    filter[1] = "pass";
    min[2] = min_value;
    max[2] = 255;
    filter[2] = "stop";
    for (i = 0; i < 3; i++) {
        selectWindow("" + i);
        setThreshold(min[i], max[i]);
        run("Convert to Mask");
        if (filter[i] == "stop") run("Invert");
    }
    imageCalculator("AND create", "0", "1");
    imageCalculator("AND create", "Result of 0", "2");
    for (i = 0; i < 3; i++) {
        selectWindow("" + i);
        close();
    }
    selectWindow("Result of 0");
    close();
    selectWindow("Result of Result of 0");
    rename(a);

    // Turn the thresholded image into a binary image
    setOption("BlackBackground", true);
    run("Make Binary");
    resetThreshold();

    // Create a selection = selecting the area (ROI) that will be measured
    run("Create Selection");

    // Transfer the selection from the "Mask" image to the original image
    selectImage(currentImage);
    run("Restore Selection");

    // Blackout and clear the background
    setBackgroundColor(0, 0, 0);
    run("Clear Outside");
    saveAs("Jpeg", input + filename);

    // Duplicate masked image to make 3 copies that will be used for extracting RGB, Lab, and HSB values
    run("Duplicate...", "title=RGB");
    rename(filename + "_RGB");

    run("Duplicate...", "title=LAB");
    rename(filename + "_LAB");

    run("Duplicate...", "title=HSB");
    rename(filename + "_HSB");

    // Break down the image into its composite channels
    selectWindow(filename + "_RGB");
    run("Make Composite");
    run("Restore Selection");
    roiManager("Add");

    // Set the measurements in the ROI manager
    run("Set Measurements...", "area display mean standard modal min limit redirect=None decimal=3");

    // Use multimeasure to collect these data from each of the 3 composite channels
    roiManager("multi-measure measure_all one append");

    // Clear ROI manager and Mask image
    roiManager("Deselect");
    roiManager("Delete");
    close(filename + "_RGB");

    // Converts image pixels from sRGB color space to CIE L*a*b* color space
    selectWindow(filename + "_LAB");
    run("Lab Stack");
    run("Restore Selection");
    roiManager("Add");

    // Set the measurements in the ROI manager
    run("Set Measurements...", "area display mean standard modal min limit redirect=None decimal=3");

    // Use multimeasure to collect these data from each of the 3 composite channels
    roiManager("multi-measure measure_all one append");

    // Clear ROI manager and Mask image
    roiManager("Deselect");
    roiManager("Delete");
    close(filename + "_LAB");

    // Converts image pixels from sRGB color space to HSB color space
    selectWindow(filename + "_HSB");
    run("HSB Stack");
    run("Restore Selection");
    roiManager("Add");

    // Set the measurements in the ROI manager
    run("Set Measurements...", "area display mean standard modal min limit redirect=None decimal=3");

    // Use multimeasure to collect these data from each of the 3 composite channels
    roiManager("multi-measure measure_all one append");

    // Clear ROI manager and Mask image
    roiManager("Deselect");
    roiManager("Delete");
    close("Mask");
    close(filename + "_HSB");

    // Save data collected by ROI manager to a .csv file titled "Whole_Color_Measurements.csv"
    dir = getDirectory("image");
    name = "Whole_Color_Measurements";
    index = lastIndexOf(name, "\\");
    if (index != -1) name = substring(name, 0, index);
    name = name + ".csv";
    saveAs("Measurements", dir + name);

    close();
}

run("Clear Results");
