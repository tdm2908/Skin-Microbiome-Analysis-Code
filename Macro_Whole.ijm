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

    // Get average intensity of the whole image
    run("Measure");

    // Save results to a .csv file titled "Whole_Color_Measurements.csv"
    dir = getDirectory("image");
    name = "Whole_Color_Measurements";
    index = lastIndexOf(name, "\\");
    if (index != -1) name = substring(name, 0, index);
    name = name + ".csv";
    saveAs("Results", dir + name);

    // Close the image
    close();
}

run("Clear Results");
